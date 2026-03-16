"""Stage 1-4 パイプラインのユニットテスト."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sidecar.models import Analysis, CoachingContext, DeathScene, GoodPlay, Observation
from sidecar.pipeline import (
    _cache_key,
    context_to_system_instruction,
    load_cached_context,
    run_analyze,
    run_stage1_observer,
    run_stage3_analyst,
    run_stage4_coach,
    save_context,
)


_VALID_STRUCTURED_PAYLOAD: dict[str, object] = {
    "deaths": [
        {
            "time": "15:41",
            "cause": "マルファイト ULT",
            "root_cause": "positioning_failure",
            "actor_champion": "Malphite",
            "note": "立ち位置が甘い",
            "direct_answer": "川側の視界がないのに engage 圏へ先に入ったのが死因",
            "improvement_answer": "味方の寄りを 1 秒待つか先に ward を置けば防げた",
            "evidence": "15:38 に川ブッシュ未確認のまま前進",
            "counterfactual": "ADC の斜め後ろで R 確認まで待つ",
            "coach_rule": "未視認の engage ult がある時は一列後ろで構える",
            "replay_window": "15:38-15:47",
            "visual_focus": "川側ブッシュとの距離",
        }
    ],
    "good_plays": [
        {
            "time": "08:39",
            "note": "味方が捕まった瞬間にヒールを合わせた",
            "direct_answer": "敵の動きを先読みして最速でヒールを合わせた",
            "trigger": "味方 ADC が CC 被弾",
            "read": "Nami の bubble 発生と同時に敵が前に出た",
            "action": "heal 即発動 → 自身は後退",
            "timing_window": "CC ヒット後 0.3 秒以内",
            "reusable_rule": "味方 CC を見たら即 heal + 後退がセット",
            "replay_window": "08:36-08:44",
            "visual_focus": "heal 発動タイミングと自身の足位置",
        }
    ],
}


def make_observation() -> Observation:
    """テスト用の Observation を返す."""
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
            GoodPlay(
                time="18:39",
                note="味方 CC 直後にヒールを先読みした",
                direct_answer="CC 後の追撃タイミングを読んで先読みヒールを打った",
                trigger="味方が stun を受けた",
                read="敵が前進モーションを開始",
                action="heal を stun 終了前に先打ち",
                timing_window="stun 残り 0.5 秒前",
                reusable_rule="敵の前進モーションが見えたら heal 先打ちで間に合う",
                replay_window="18:36-18:44",
                visual_focus="heal のタイミングと敵の前進位置",
            ),
        ),
    )


class TestStage1Observer:
    """Stage 1: Observer のテスト."""

    def test_returns_observation(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        video_path = tmp_path / "match.mp4"
        video_path.write_bytes(b"fake")
        observed_calls: list[str] = []

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setattr("sidecar.pipeline._get_gemini_client", lambda: object())
        monkeypatch.setattr(
            "sidecar.pipeline._upload_video_file",
            lambda client, path: observed_calls.append(f"upload:{path.name}") or object(),
        )
        monkeypatch.setattr(
            "sidecar.pipeline._wait_for_uploaded_file",
            lambda client, uploaded_file: observed_calls.append("wait") or uploaded_file,
        )
        monkeypatch.setattr(
            "sidecar.pipeline._run_stage1_overview_pass",
            lambda client, uploaded_file: observed_calls.append("pass1")
            or [{"timestamp": "08:39", "scene_type": "good_play", "short_reason": "ヒール"}],
        )
        monkeypatch.setattr(
            "sidecar.pipeline._run_stage1_detail_pass",
            lambda client, uploaded_file, candidates: observed_calls.append("pass2")
            or {"summary": "良いプレイ候補あり"},
        )
        monkeypatch.setattr(
            "sidecar.pipeline._run_stage1_structured_pass",
            lambda client, overview, detail: observed_calls.append("pass3")
            or _VALID_STRUCTURED_PAYLOAD,
        )
        monkeypatch.setattr(
            "sidecar.pipeline._delete_uploaded_file",
            lambda client, uploaded_file: observed_calls.append("delete"),
        )

        result = run_stage1_observer(str(video_path))

        assert isinstance(result, Observation)
        assert observed_calls == ["upload:match.mp4", "wait", "pass1", "pass2", "pass3", "delete"]

    def test_observation_is_immutable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        video_path = tmp_path / "match.mp4"
        video_path.write_bytes(b"fake")
        self._install_success_mocks(monkeypatch)

        result = run_stage1_observer(str(video_path))

        with pytest.raises(Exception):
            result.deaths = ()  # type: ignore[misc]

    def test_rejects_empty_video_path(self) -> None:
        with pytest.raises(ValueError, match="video_path"):
            run_stage1_observer("   ")

    def test_rejects_missing_video_path(self, tmp_path: Path) -> None:
        missing_path = tmp_path / "missing.mp4"

        with pytest.raises(FileNotFoundError):
            run_stage1_observer(str(missing_path))

    def test_rejects_non_mp4_file(self, tmp_path: Path) -> None:
        text_path = tmp_path / "match.txt"
        text_path.write_text("fake")

        with pytest.raises(ValueError, match="MP4"):
            run_stage1_observer(str(text_path))

    def test_rejects_missing_api_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        video_path = tmp_path / "match.mp4"
        video_path.write_bytes(b"fake")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            run_stage1_observer(str(video_path))

    def test_rejects_schema_mismatch(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        video_path = tmp_path / "match.mp4"
        video_path.write_bytes(b"fake")
        self._install_success_mocks(
            monkeypatch,
            structured_payload={"deaths": [{"time": "15:41", "cause": "原因のみ"}], "good_plays": []},
        )

        with pytest.raises(ValueError, match="schema"):
            run_stage1_observer(str(video_path))

    def test_deletes_uploaded_file_when_pass_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        video_path = tmp_path / "match.mp4"
        video_path.write_bytes(b"fake")
        observed_calls: list[str] = []

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setattr("sidecar.pipeline._get_gemini_client", lambda: object())
        monkeypatch.setattr("sidecar.pipeline._upload_video_file", lambda client, path: object())
        monkeypatch.setattr(
            "sidecar.pipeline._wait_for_uploaded_file",
            lambda client, uploaded_file: uploaded_file,
        )
        monkeypatch.setattr(
            "sidecar.pipeline._run_stage1_overview_pass",
            lambda client, uploaded_file: (_ for _ in ()).throw(RuntimeError("pass1 failed")),
        )
        monkeypatch.setattr(
            "sidecar.pipeline._delete_uploaded_file",
            lambda client, uploaded_file: observed_calls.append("delete"),
        )

        with pytest.raises(RuntimeError, match="pass1 failed"):
            run_stage1_observer(str(video_path))

        assert observed_calls == ["delete"]

    @staticmethod
    def _install_success_mocks(
        monkeypatch: pytest.MonkeyPatch,
        *,
        structured_payload: dict[str, object] | None = None,
    ) -> None:
        payload = structured_payload or _VALID_STRUCTURED_PAYLOAD
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setattr("sidecar.pipeline._get_gemini_client", lambda: object())
        monkeypatch.setattr("sidecar.pipeline._upload_video_file", lambda client, path: object())
        monkeypatch.setattr(
            "sidecar.pipeline._wait_for_uploaded_file",
            lambda client, uploaded_file: uploaded_file,
        )
        monkeypatch.setattr(
            "sidecar.pipeline._run_stage1_overview_pass",
            lambda client, uploaded_file: [{"timestamp": "08:39", "scene_type": "good_play", "short_reason": "ヒール"}],
        )
        monkeypatch.setattr(
            "sidecar.pipeline._run_stage1_detail_pass",
            lambda client, uploaded_file, candidates: {"summary": "良いプレイ候補あり"},
        )
        monkeypatch.setattr(
            "sidecar.pipeline._run_stage1_structured_pass",
            lambda client, overview, detail: payload,
        )
        monkeypatch.setattr("sidecar.pipeline._delete_uploaded_file", lambda client, uploaded_file: None)


class TestStage3Analyst:
    """Stage 3: Analyst のテスト."""

    def test_returns_analysis(self) -> None:
        result = run_stage3_analyst(make_observation())
        assert isinstance(result, Analysis)

    def test_weakness_count_accurate(self) -> None:
        obs = make_observation()
        result = run_stage3_analyst(obs)
        assert result.total_deaths == len(obs.deaths)

    def test_good_scenes_match_good_plays(self) -> None:
        obs = make_observation()
        result = run_stage3_analyst(obs)
        assert result.good_count == len(obs.good_plays)

    def test_main_weakness_cause_is_most_frequent_root_cause(self) -> None:
        obs = make_observation()
        result = run_stage3_analyst(obs)
        root_causes = [death.root_cause for death in obs.deaths]
        assert result.main_weakness_cause == max(set(root_causes), key=root_causes.count)

    def test_weakness_contains_death_count(self) -> None:
        obs = make_observation()
        result = run_stage3_analyst(obs)
        assert str(result.total_deaths) in result.weakness or result.total_deaths == 0


class TestStage4Coach:
    """Stage 4: Coach のテスト."""

    def test_returns_string(self) -> None:
        obs = make_observation()
        result = run_stage4_coach(run_stage3_analyst(obs), obs)
        assert isinstance(result, str)

    def test_contains_tool_usage_policy(self) -> None:
        obs = make_observation()
        result = run_stage4_coach(run_stage3_analyst(obs), obs)
        assert "seek_replay" in result

    def test_contains_seek_and_pause_tools(self) -> None:
        obs = make_observation()
        result = run_stage4_coach(run_stage3_analyst(obs), obs)
        assert "seek_replay" in result
        assert "pause_replay" in result

    def test_contains_slow_motion_tool(self) -> None:
        obs = make_observation()
        result = run_stage4_coach(run_stage3_analyst(obs), obs)
        assert "slow_motion" in result

    def test_not_empty(self) -> None:
        obs = make_observation()
        result = run_stage4_coach(run_stage3_analyst(obs), obs)
        assert len(result) > 80

    def test_contains_scene_card_direct_answer(self) -> None:
        obs = make_observation()
        result = run_stage4_coach(run_stage3_analyst(obs), obs)
        assert "川側の視界がないのに engage 圏へ先に入ったのが死因" in result

    def test_contains_good_play_rule(self) -> None:
        obs = make_observation()
        result = run_stage4_coach(run_stage3_analyst(obs), obs)
        assert "味方 CC を見たら即 heal + 後退がセット" in result


class TestRunAnalyze:
    """run_analyze 統合テスト（Stage A→B→C アーキテクチャ）.

    新アーキテクチャでは run_analyze が内部で fetch_match_data を呼ぶため、
    Riot API + Gemini の両方をモックする必要がある。
    ここでは Stage 3/4（Synthesis）の動作をテストする。
    Stage B（Vision Analysis）は _analyze_single_scene を直接モックする。
    """

    @staticmethod
    def _install_analyze_mocks(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """run_analyze に必要なモックをまとめて設定する."""
        from sidecar.models import MatchInfo, Participant
        from sidecar.riot_api import SceneCandidate

        match_info = MatchInfo(
            match_id="JP1_TEST",
            classification="full",
            participants=(
                Participant(champion="Lux", role="BOTTOM", team="ally"),
                Participant(champion="Zed", role="MIDDLE", team="enemy"),
            ),
            player_champion="Lux",
        )
        candidates = [
            SceneCandidate(
                scene_id="death_000",
                scene_type="death",
                match_time_ms=341_000,
                actor_champion="Zed",
                victim_champion="Lux",
                assist_champions=(),
                clip_start_ms=329_000,
                clip_end_ms=349_000,
                phase="mid",
                map_zone="lane",
                gold_diff_at_time=0,
                importance_score=3,
                objective_context="",
                bounty=0,
                fight_context="solo",
                death_cost_score=0,
                kill_value_score=0,
            ),
        ]

        from sidecar.models import MatchMeta, RawMatchData

        raw_match_data = RawMatchData(
            meta=MatchMeta(
                match_id="JP1_TEST",
                game_duration_ms=2100000,
                win=True,
                player_champion="Lux",
                player_role="BOTTOM",
            ),
            participants=(),
            kill_events=(),
            objective_events=(),
            building_events=(),
            ward_events=(),
        )

        monkeypatch.setattr(
            "sidecar.riot_api.fetch_match_data",
            lambda _mid: (match_info, candidates, raw_match_data),
        )
        monkeypatch.setattr("sidecar.pipeline.SESSIONS_DIR", tmp_path / "sessions")
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setattr("sidecar.pipeline._get_gemini_client", lambda: object())
        monkeypatch.setattr("sidecar.pipeline._upload_video_file", lambda c, p: object())
        monkeypatch.setattr("sidecar.pipeline._wait_for_uploaded_file", lambda c, f: f)
        monkeypatch.setattr("sidecar.pipeline._delete_uploaded_file", lambda c, f: None)

        # _analyze_single_scene をモックして DeathScene を返す
        def fake_analyze_scene(_client, _file, candidate, _prompt, _ctx):
            if candidate.scene_type == "death":
                return DeathScene(
                    time="05:41",
                    cause="Zed による キル",
                    note="ミッドで孤立",
                    root_cause="positioning_failure",
                    actor_champion="Zed",
                    direct_answer="視界なしで前に出すぎた",
                    improvement_answer="味方と合流してから動く",
                    evidence="ワードなしで川に侵入",
                    counterfactual="タワー下で待つ",
                    coach_rule="視界のない場所で孤立しない",
                    replay_window="05:29-05:49",
                    visual_focus="ミニマップの暗い領域",
                )
            return None

        monkeypatch.setattr("sidecar.pipeline._analyze_single_scene", fake_analyze_scene)

    def test_returns_coaching_context(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        video_path = tmp_path / "match.mp4"
        video_path.write_bytes(b"fake")
        self._install_analyze_mocks(monkeypatch, tmp_path)

        result, saved_path = run_analyze(str(video_path), "JP1_TEST")

        assert isinstance(result, CoachingContext)
        assert saved_path.exists()

    def test_result_has_non_empty_system_instruction(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        video_path = tmp_path / "match.mp4"
        video_path.write_bytes(b"fake")
        self._install_analyze_mocks(monkeypatch, tmp_path)

        result, _path = run_analyze(str(video_path), "JP1_TEST")

        assert len(result.system_instruction) > 80

    def test_context_to_system_instruction_returns_system_instruction(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        video_path = tmp_path / "match.mp4"
        video_path.write_bytes(b"fake")
        self._install_analyze_mocks(monkeypatch, tmp_path)

        ctx, _path = run_analyze(str(video_path), "JP1_TEST")

        assert context_to_system_instruction(ctx) == ctx.system_instruction

    def test_result_contains_match_info(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        video_path = tmp_path / "match.mp4"
        video_path.write_bytes(b"fake")
        self._install_analyze_mocks(monkeypatch, tmp_path)

        ctx, _path = run_analyze(str(video_path), "JP1_TEST")

        assert ctx.match_info.match_id == "JP1_TEST"
        assert ctx.match_info.classification == "full"


class TestCacheLogic:
    """キャッシュ層のテスト."""

    def test_save_and_load_roundtrip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        video_path = tmp_path / "match.mp4"
        video_path.write_bytes(b"fake")
        monkeypatch.setattr("sidecar.pipeline.SESSIONS_DIR", tmp_path / "sessions")

        obs = make_observation()
        analysis = run_stage3_analyst(obs)
        system_instruction = run_stage4_coach(analysis, obs)
        from sidecar.models import MatchInfo
        ctx = CoachingContext.from_observation_analysis(
            video_source="match.mp4",
            match_info=MatchInfo.unavailable(),
            obs=obs,
            analysis=analysis,
            system_instruction=system_instruction,
        )

        saved_path = save_context(ctx, str(video_path))
        loaded = load_cached_context(str(video_path))

        assert saved_path.exists()
        assert loaded is not None
        assert loaded == ctx

    def test_cache_miss_returns_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        video_path = tmp_path / "match.mp4"
        video_path.write_bytes(b"fake")
        monkeypatch.setattr("sidecar.pipeline.SESSIONS_DIR", tmp_path / "sessions")

        result = load_cached_context(str(video_path))

        assert result is None

    def test_corrupt_cache_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        video_path = tmp_path / "match.mp4"
        video_path.write_bytes(b"fake")
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        monkeypatch.setattr("sidecar.pipeline.SESSIONS_DIR", sessions_dir)

        # キャッシュキーと同じファイル名で壊れた JSON を作成
        key = _cache_key(str(video_path))
        (sessions_dir / f"{key}.json").write_text("not valid json")

        result = load_cached_context(str(video_path))

        assert result is None

    def test_wrong_schema_version_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        video_path = tmp_path / "match.mp4"
        video_path.write_bytes(b"fake")
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        monkeypatch.setattr("sidecar.pipeline.SESSIONS_DIR", sessions_dir)

        key = _cache_key(str(video_path))
        bad_data = {"schema_version": "0.9", "deaths": [], "good_plays": []}
        (sessions_dir / f"{key}.json").write_text(json.dumps(bad_data))

        result = load_cached_context(str(video_path))

        assert result is None

    def test_cache_roundtrip_with_fingerprint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """riot_fingerprint 付きキャッシュが正しく保存・復元される."""
        video_path = tmp_path / "match.mp4"
        video_path.write_bytes(b"fake")
        monkeypatch.setattr("sidecar.pipeline.SESSIONS_DIR", tmp_path / "sessions")

        from sidecar.models import MatchInfo
        obs = make_observation()
        analysis = run_stage3_analyst(obs)
        system_instruction = run_stage4_coach(analysis, obs)
        ctx = CoachingContext.from_observation_analysis(
            video_source="match.mp4",
            match_info=MatchInfo.unavailable(),
            obs=obs,
            analysis=analysis,
            system_instruction=system_instruction,
        )

        fp = "test_fingerprint"
        save_context(ctx, str(video_path), fp)
        loaded = load_cached_context(str(video_path), fp)
        miss = load_cached_context(str(video_path), "different_fp")

        assert loaded == ctx
        assert miss is None
