"""FastAPI sidecar エンドポイントのテスト."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress

import pytest
import pytest_asyncio

from sidecar import main
from sidecar.models import CoachState, DialogueState, SubtitleEvent


@pytest_asyncio.fixture(autouse=True)
async def reset_app_state() -> None:
    """各テスト前後でグローバル状態を初期化する."""
    main._coach_state = CoachState.IDLE
    main._dialogue_state = DialogueState.IDLE
    main._latest_subtitle = None
    main._latest_ai_subtitle = None
    main._latest_user_subtitle = None
    main._engine = None
    main._session_task = None
    main._analyze_task = None
    main._context_path = None

    yield

    if main._engine is not None:
        with suppress(Exception):
            await main._engine.stop()
        main._engine = None

    for task_attr in ("_session_task", "_analyze_task"):
        task = getattr(main, task_attr, None)
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        setattr(main, task_attr, None)

    main._coach_state = CoachState.IDLE
    main._dialogue_state = DialogueState.IDLE
    main._latest_subtitle = None
    main._latest_ai_subtitle = None
    main._latest_user_subtitle = None
    main._engine = None
    main._context_path = None


class TestHealthEndpoint:
    """GET /health のテスト."""

    @pytest.mark.asyncio
    async def test_status_ok(self) -> None:
        res = await main.health()
        assert res["status"] == "ok"

    @pytest.mark.asyncio
    async def test_state_field_exists(self) -> None:
        res = await main.health()
        assert "state" in res


class TestStateEndpoint:
    """GET /state のテスト."""

    @pytest.mark.asyncio
    async def test_returns_coach_and_dialogue_state(self) -> None:
        res = await main.get_state()
        assert "coach_state" in res
        assert "dialogue_state" in res

    @pytest.mark.asyncio
    async def test_initial_state_is_idle(self) -> None:
        res = await main.get_state()
        assert res["coach_state"] == "idle"
        assert res["dialogue_state"] == "idle"

    @pytest.mark.asyncio
    async def test_includes_context_path_when_ready(self) -> None:
        main._coach_state = CoachState.READY
        main._context_path = "data/sessions/abc.json"

        res = await main.get_state()

        assert res["coach_state"] == "ready"
        assert res["context_path"] == "data/sessions/abc.json"

    @pytest.mark.asyncio
    async def test_excludes_context_path_when_not_ready(self) -> None:
        main._coach_state = CoachState.IDLE
        main._context_path = None

        res = await main.get_state()

        assert "context_path" not in res


class TestAccessLogFilter:
    """高頻度ポーリング用アクセスログの抑止を確認する."""

    @staticmethod
    def _make_access_record(path: str) -> logging.LogRecord:
        return logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg='%s - "%s %s HTTP/%s" %d',
            args=("127.0.0.1:12345", "GET", path, "1.1", 200),
            exc_info=None,
        )

    def test_filter_suppresses_polling_endpoints(self) -> None:
        log_filter = main.PollingAccessLogFilter()

        assert not log_filter.filter(self._make_access_record("/health"))
        assert not log_filter.filter(self._make_access_record("/state"))
        assert not log_filter.filter(self._make_access_record("/subtitles"))

    def test_filter_keeps_non_polling_endpoints(self) -> None:
        log_filter = main.PollingAccessLogFilter()

        assert log_filter.filter(self._make_access_record("/start"))
        assert log_filter.filter(self._make_access_record("/ptt/start"))

    def test_configure_access_log_filter_is_idempotent(self) -> None:
        logger = logging.getLogger("tests.uvicorn.access")
        original_filters = list(logger.filters)
        logger.filters = []

        try:
            main._configure_access_log_filter(logger)
            main._configure_access_log_filter(logger)

            installed = [
                log_filter
                for log_filter in logger.filters
                if isinstance(log_filter, main.PollingAccessLogFilter)
            ]
            assert len(installed) == 1
        finally:
            logger.filters = original_filters


class TestSubtitlesEndpoint:
    """GET /subtitles のテスト."""

    @pytest.mark.asyncio
    async def test_returns_subtitle_fields(self) -> None:
        res = await main.get_subtitles()
        assert "text" in res
        assert "timestamp" in res
        assert "is_user" in res

    @pytest.mark.asyncio
    async def test_empty_when_no_subtitle(self) -> None:
        res = await main.get_subtitles()
        assert res["text"] == ""


class TestSubtitleAggregation:
    """字幕の断片結合を確認する."""

    def test_merges_same_speaker_fragments_without_duplication(self) -> None:
        main._on_subtitle(
            SubtitleEvent(text="途中で割り込", timestamp=10.0, is_user=True)
        )
        main._on_subtitle(
            SubtitleEvent(text="途中で割り込みできる", timestamp=10.4, is_user=True)
        )

        assert main._latest_subtitle is not None
        assert main._latest_subtitle.text == "途中で割り込みできる"

    def test_merges_same_speaker_disjoint_fragments(self) -> None:
        main._on_subtitle(SubtitleEvent(text="へえ、そう", timestamp=20.0, is_user=False))
        main._on_subtitle(SubtitleEvent(text="なんだ!", timestamp=20.3, is_user=False))

        assert main._latest_subtitle is not None
        assert main._latest_subtitle.text == "へえ、そうなんだ!"

    def test_replaces_subtitle_when_speaker_changes(self) -> None:
        main._on_subtitle(SubtitleEvent(text="こんにちは", timestamp=30.0, is_user=True))
        main._on_subtitle(SubtitleEvent(text="やあ", timestamp=30.2, is_user=False))

        assert main._latest_subtitle is not None
        assert main._latest_subtitle.text == "やあ"
        assert not main._latest_subtitle.is_user

    def test_replaces_subtitle_when_gap_is_large(self) -> None:
        main._on_subtitle(SubtitleEvent(text="最初", timestamp=40.0, is_user=False))
        main._on_subtitle(SubtitleEvent(text="次の文", timestamp=42.0, is_user=False))

        assert main._latest_subtitle is not None
        assert main._latest_subtitle.text == "次の文"


class TestAnalyzeEndpoint:
    """POST /analyze のテスト."""

    @pytest.mark.asyncio
    async def test_returns_analyzing_and_changes_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        released = asyncio.Event()

        async def fake_analyze_session(_video_path: str, _match_id: str = "") -> None:
            await released.wait()

        monkeypatch.setattr(main, "_run_analyze_session", fake_analyze_session)
        request = main.AnalyzeRequest(video_path="match.mp4")

        res = await main.analyze_video(request)

        assert res["status"] == "analyzing"
        assert main._coach_state == CoachState.ANALYZING

        released.set()
        if main._analyze_task is not None:
            await asyncio.wait_for(main._analyze_task, timeout=1.0)

    @pytest.mark.asyncio
    async def test_returns_busy_when_not_idle(self) -> None:
        main._coach_state = CoachState.ANALYZING
        request = main.AnalyzeRequest(video_path="match.mp4")

        res = await main.analyze_video(request)

        assert res["status"] == "busy"

    @pytest.mark.asyncio
    async def test_immediate_return_does_not_block(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """POST /analyze が即時返却されること（event loop をブロックしない）."""
        blocker = asyncio.Event()

        async def fake_analyze_session(_video_path: str, _match_id: str = "") -> None:
            await blocker.wait()

        monkeypatch.setattr(main, "_run_analyze_session", fake_analyze_session)
        request = main.AnalyzeRequest(video_path="match.mp4")

        # タイムアウト 0.5 秒以内に返ること
        res = await asyncio.wait_for(main.analyze_video(request), timeout=0.5)
        assert res["status"] == "analyzing"

        blocker.set()
        if main._analyze_task is not None:
            await asyncio.wait_for(main._analyze_task, timeout=1.0)


class TestStartEndpoint:
    """POST /start のテスト."""

    @pytest.mark.asyncio
    async def test_returns_not_ready_when_not_ready(self) -> None:
        request = main.StartSessionRequest(context_path="data/sessions/abc.json")

        res = await main.start_session(request)

        assert res["status"] == "not_ready"

    @pytest.mark.asyncio
    async def test_returns_started_when_ready(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        released = asyncio.Event()

        async def fake_live_session(_context_path: str) -> None:
            await released.wait()

        monkeypatch.setattr(main, "_run_live_session", fake_live_session)
        main._coach_state = CoachState.READY
        main._context_path = "data/sessions/abc.json"
        request = main.StartSessionRequest(context_path="data/sessions/abc.json")

        res = await main.start_session(request)

        assert res["status"] == "started"
        assert main._coach_state == CoachState.DIALOGUE

        released.set()
        if main._session_task is not None:
            await asyncio.wait_for(main._session_task, timeout=1.0)

    @pytest.mark.asyncio
    async def test_passes_context_path_to_live_session(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        observed: list[str] = []
        released = asyncio.Event()

        async def fake_live_session(context_path: str) -> None:
            observed.append(context_path)
            await released.wait()

        monkeypatch.setattr(main, "_run_live_session", fake_live_session)
        main._coach_state = CoachState.READY
        request = main.StartSessionRequest(context_path="data/sessions/xyz.json")

        await main.start_session(request)
        await asyncio.wait_for(asyncio.sleep(0), timeout=1.0)

        assert observed == ["data/sessions/xyz.json"]

        released.set()
        if main._session_task is not None:
            await asyncio.wait_for(main._session_task, timeout=1.0)


class TestStopEndpoint:
    """POST /stop のテスト."""

    @pytest.mark.asyncio
    async def test_not_running_when_idle(self) -> None:
        res = await main.stop_session()
        assert res["status"] == "not_running"

    @pytest.mark.asyncio
    async def test_stop_waits_for_engine_and_session_task(
        self,
    ) -> None:
        session_released = asyncio.Event()

        class DummyEngine:
            def __init__(self) -> None:
                self.stop_called = False

            async def stop(self) -> None:
                self.stop_called = True
                session_released.set()

        async def fake_session() -> None:
            await session_released.wait()

        engine = DummyEngine()
        main._coach_state = CoachState.DIALOGUE
        main._dialogue_state = DialogueState.SPEAKING
        main._engine = engine
        main._session_task = asyncio.create_task(fake_session())

        res = await main.stop_session()

        assert res["status"] == "stopped"
        assert engine.stop_called
        assert main._session_task is None
        assert main._coach_state == CoachState.IDLE
        assert main._dialogue_state == DialogueState.IDLE

    @pytest.mark.asyncio
    async def test_stop_cancels_analyze_task(self) -> None:
        async def fake_analyze() -> None:
            await asyncio.Event().wait()

        main._coach_state = CoachState.ANALYZING
        task = asyncio.create_task(fake_analyze())
        main._analyze_task = task

        res = await main.stop_session()

        assert res["status"] == "stopped"
        assert task.cancelled()
        assert main._analyze_task is None

    @pytest.mark.asyncio
    async def test_stop_clears_context_path(self) -> None:
        main._coach_state = CoachState.READY
        main._context_path = "data/sessions/abc.json"

        res = await main.stop_session()

        assert res["status"] == "stopped"
        assert main._context_path is None


class TestRunAnalyzeSession:
    """_run_analyze_session のテスト."""

    @pytest.mark.asyncio
    async def test_transitions_to_ready_on_success(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pytest.MonkeyPatch,
    ) -> None:
        from sidecar.models import CoachingContext
        from sidecar.pipeline import run_stage3_analyst, run_stage4_coach

        obs = make_observation()
        analysis = run_stage3_analyst(obs)
        from sidecar.models import MatchInfo
        ctx = CoachingContext.from_observation_analysis(
            video_source="match.mp4",
            match_info=MatchInfo.unavailable(),
            obs=obs,
            analysis=analysis,
            system_instruction=run_stage4_coach(analysis, obs),
        )

        from pathlib import Path as _Path
        fake_path = _Path("/tmp/test_ctx.json")
        monkeypatch.setattr(main, "run_analyze", lambda _vp, _mid="": (ctx, fake_path))

        await main._run_analyze_session("match.mp4")

        assert main._coach_state == CoachState.READY
        assert main._context_path == str(fake_path)

    @pytest.mark.asyncio
    async def test_transitions_to_error_on_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fail(_vp: str) -> None:
            raise RuntimeError("分析失敗")

        monkeypatch.setattr(main, "run_analyze", fail)

        await main._run_analyze_session("match.mp4")

        assert main._coach_state == CoachState.ERROR


def make_observation() -> object:
    """テスト用 Observation を返す（test_main 内用）."""
    from sidecar.models import DeathScene, GoodPlay, Observation

    return Observation(
        deaths=(
            DeathScene(
                time="15:41",
                cause="マルファイト ULT",
                note="ULT 射程内に立ち続けた",
                root_cause="positioning_failure",
                actor_champion="Malphite",
                direct_answer="川側の視界がないのに engage 圏へ先に入ったのが死因",
                improvement_answer="味方の寄りを 1 秒待つか先に ward を置けば防げた",
                evidence="15:38 に川ブッシュ未確認のまま前進",
                counterfactual="ADC の斜め後ろで R 確認まで待つ",
                coach_rule="未視認の engage ult がある時は一列後ろで構える",
                replay_window="15:38-15:47",
                visual_focus="川側ブッシュとの距離",
            ),
            DeathScene(
                time="17:39",
                cause="マルファイト ULT",
                note="壁際なら範囲外だった",
                root_cause="positioning_failure",
                actor_champion="Malphite",
                direct_answer="壁から離れた位置で静止して R の射程に入った",
                improvement_answer="壁際に密着して R の AoE から外れる",
                evidence="17:36 に壁から離れた位置で静止",
                counterfactual="壁際に張り付いて R の射程を切る",
                coach_rule="Malphite R 警戒時は壁際に密着して R の射程を切る",
                replay_window="17:36-17:42",
                visual_focus="壁との距離と R の着弾範囲",
            ),
        ),
        good_plays=(
            GoodPlay(
                time="08:39",
                note="味方が捕まった瞬間にヒールを合わせた",
                direct_answer="敵の動きを先読みして最速でヒールを合わせた",
                trigger="味方 ADC が CC 被弾",
                read="Nami の bubble 発生と同時に敵が前に出た",
                action="heal 即発動 → 自身は後退",
                timing_window="CC ヒット後 0.3 秒以内",
                reusable_rule="味方 CC を見たら即 heal + 後退がセット",
                replay_window="08:36-08:44",
                visual_focus="heal 発動タイミングと自身の足位置",
            ),
        ),
    )


class TestRunLiveSession:
    """_run_live_session のテスト."""

    @pytest.mark.asyncio
    async def test_transitions_to_idle_after_completion(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pytest.MonkeyPatch,
    ) -> None:
        from pathlib import Path

        from sidecar.models import CoachingContext
        from sidecar.pipeline import run_stage3_analyst, run_stage4_coach

        obs = make_observation()
        analysis = run_stage3_analyst(obs)
        from sidecar.models import MatchInfo
        ctx = CoachingContext.from_observation_analysis(
            video_source="match.mp4",
            match_info=MatchInfo.unavailable(),
            obs=obs,
            analysis=analysis,
            system_instruction=run_stage4_coach(analysis, obs),
        )

        sessions_dir = Path("data/sessions")
        sessions_dir.mkdir(parents=True, exist_ok=True)
        context_file = sessions_dir / "test_ctx.json"
        context_file.write_text(ctx.to_json(), encoding="utf-8")

        class DummyEngine:
            def __init__(self, **kwargs: object) -> None:
                pass

            async def start(self) -> None:
                return None

        monkeypatch.setattr(main, "RealtimeEngine", DummyEngine)
        main._coach_state = CoachState.DIALOGUE

        await main._run_live_session(str(context_file))

        assert main._coach_state == CoachState.IDLE

    @pytest.mark.asyncio
    async def test_rejects_path_outside_sessions_dir(self) -> None:
        """data/sessions/ 外のパスを拒否する."""
        main._coach_state = CoachState.DIALOGUE
        await main._run_live_session("/tmp/malicious.json")
        assert main._coach_state == CoachState.ERROR

    @pytest.mark.asyncio
    async def test_rejects_sessions_prefix_dir(self) -> None:
        """data/sessions2/ のような prefix 一致ディレクトリを拒否する."""
        from pathlib import Path

        fake_dir = Path("data/sessions2")
        fake_dir.mkdir(parents=True, exist_ok=True)
        fake_file = fake_dir / "trick.json"
        fake_file.write_text("{}", encoding="utf-8")

        main._coach_state = CoachState.DIALOGUE
        await main._run_live_session(str(fake_file))
        assert main._coach_state == CoachState.ERROR

        # クリーンアップ
        fake_file.unlink()
        fake_dir.rmdir()

    @pytest.mark.asyncio
    async def test_rejects_non_json_file(self) -> None:
        """data/sessions/ 配下でも .json 以外を拒否する."""
        from pathlib import Path

        sessions_dir = Path("data/sessions")
        sessions_dir.mkdir(parents=True, exist_ok=True)
        txt_file = sessions_dir / "test.txt"
        txt_file.write_text("not json", encoding="utf-8")

        main._coach_state = CoachState.DIALOGUE
        await main._run_live_session(str(txt_file))
        assert main._coach_state == CoachState.ERROR

        txt_file.unlink()

    @pytest.mark.asyncio
    async def test_rejects_traversal_with_dotdot(self) -> None:
        """.. を含むパスを拒否する."""
        main._coach_state = CoachState.DIALOGUE
        await main._run_live_session("data/sessions/../../etc/passwd")
        assert main._coach_state == CoachState.ERROR


class TestPushToTalkEndpoints:
    """Push-to-talk API のテスト."""

    @pytest.mark.asyncio
    async def test_start_push_to_talk_requires_running_engine(self) -> None:
        res = await main.start_push_to_talk()
        assert res["status"] == "not_running"

    @pytest.mark.asyncio
    async def test_start_push_to_talk_delegates_to_engine(self) -> None:
        class DummyEngine:
            async def start_push_to_talk(self) -> bool:
                return True

        main._coach_state = CoachState.DIALOGUE
        main._engine = DummyEngine()

        res = await main.start_push_to_talk()

        assert res["status"] == "started"

    @pytest.mark.asyncio
    async def test_end_push_to_talk_delegates_to_engine(self) -> None:
        class DummyEngine:
            async def end_push_to_talk(self) -> bool:
                return True

        main._coach_state = CoachState.DIALOGUE
        main._engine = DummyEngine()

        res = await main.end_push_to_talk()

        assert res["status"] == "ended"
