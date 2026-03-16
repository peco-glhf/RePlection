"""Stage 1-4 コーチングパイプライン."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from sidecar.models import (
    Analysis,
    CoachingContext,
    DeathScene,
    ParticipantStats,
    RawMatchData,
    GoodPlay,
    Observation,
)

logger = logging.getLogger(__name__)

MODEL_ID = "gemini-3.1-pro-preview"
FILE_READY_TIMEOUT_SECONDS = 600.0  # 10分（660MB動画の処理に数分かかる）
FILE_READY_POLL_INTERVAL_SECONDS = 5.0  # 5秒ごとにポーリング
CLIP_WINDOW_SECONDS = 15
TIME_PATTERN = re.compile(r"^\d{2}:\d{2}$")
SESSIONS_DIR = Path("data/sessions")

STAGE1_SYSTEM_PROMPT = """
あなたは League of Legends のコーチ向けアナリストです。
動画から、コーチング価値の高いデスシーンと良プレイを抽出してください。
根拠のない推測は避け、画面上で確認できる事実だけを使ってください。
""".strip()

PASS1_PROMPT = """
動画全体を俯瞰し、コーチング価値のある代表シーン候補を 5〜10 件抽出してください。
JSON のみを返してください。形式は次のとおりです。
{
  "candidates": [
    {
      "timestamp": "MM:SS",
      "scene_type": "death" または "good_play",
      "short_reason": "15文字以上60文字以内の短い理由"
    }
  ]
}
""".strip()

PASS2_PROMPT_TEMPLATE = """
以下の clip 群を詳細に見て、各候補が本当にコーチング価値を持つか評価してください。
JSON のみを返してください。形式は次のとおりです。
{{
  "details": [
    {{
      "timestamp": "MM:SS",
      "scene_type": "death" または "good_play",
      "accepted": true,
      "detail_note": "1文の詳細説明",
      "cause": "death の場合のみ短い主因。good_play では空文字可"
    }}
  ]
}}

候補一覧:
{candidate_text}
""".strip()

PASS3_PROMPT_TEMPLATE = """
以下の Pass 1 / Pass 2 の結果をもとに、最終的な Observation を JSON で返してください。
time は必ず MM:SS 形式にしてください。

フィールド定義（deaths の各要素）:
- time: タイムスタンプ（MM:SS 形式）
- cause: 即時死因（スキル名またはチャンピオン名）
- root_cause: 根本原因 ― 次の中から 1 つ選択
    positioning_failure / information_failure / cooldown_disrespect /
    strength_misjudgment / wave_timing_failure / teamfight_execution /
    greedy_play / coordination_failure
- note: 画面で確認できる事実の 1 行説明（30〜60 文字）
- direct_answer: 「なぜ死んだ？」への 1 文直接回答（40〜80 文字）
- improvement_answer: 「何を直すべき？」への 1 文回答（40〜80 文字）
- evidence: 死亡直前に画面で確認できた具体的事実（20〜50 文字）
- counterfactual: 回避できた代替行動（20〜50 文字）
- coach_rule: 次の試合で使えるルール文（30〜60 文字）
- replay_window: 推奨再生範囲 "MM:SS-MM:SS"（死亡 5 秒前〜死亡 2 秒後）
- visual_focus: 視聴者が注目すべき画面上の要素（15〜40 文字）

フィールド定義（good_plays の各要素）:
- time: タイムスタンプ（MM:SS 形式）
- note: プレイの 1 行説明（30〜60 文字）
- direct_answer: 「何が良かった？」への 1 文回答（40〜80 文字）
- trigger: このプレイを起こした起点イベント（15〜30 文字）
- read: 認識したシグナル（20〜50 文字）
- action: 実行した行動の順序（20〜50 文字）
- timing_window: 行動が有効だった時間的条件（15〜40 文字）
- reusable_rule: 同じ状況で再現できるルール（30〜60 文字）
- replay_window: 推奨再生範囲 "MM:SS-MM:SS"（プレイ 3 秒前〜プレイ 3 秒後）
- visual_focus: 視聴者が注目すべき画面上の要素（15〜40 文字）

Pass 1:
{overview_text}

Pass 2:
{detail_text}
""".strip()

_ROOT_CAUSE_ENUM = [
    "positioning_failure",
    "information_failure",
    "cooldown_disrespect",
    "strength_misjudgment",
    "wave_timing_failure",
    "teamfight_execution",
    "greedy_play",
    "coordination_failure",
]

OBSERVATION_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "deaths": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "time": {"type": "string", "maxLength": 5},
                    "cause": {"type": "string", "maxLength": 30},
                    "note": {"type": "string", "maxLength": 60},
                    "root_cause": {"type": "string", "enum": _ROOT_CAUSE_ENUM},
                    "direct_answer": {"type": "string", "maxLength": 80},
                    "improvement_answer": {"type": "string", "maxLength": 80},
                    "evidence": {"type": "string", "maxLength": 50},
                    "counterfactual": {"type": "string", "maxLength": 50},
                    "coach_rule": {"type": "string", "maxLength": 60},
                    "replay_window": {"type": "string", "maxLength": 15},
                    "visual_focus": {"type": "string", "maxLength": 40},
                },
                "required": [
                    "time", "cause", "note", "root_cause", "direct_answer",
                    "improvement_answer", "evidence", "counterfactual",
                    "coach_rule", "replay_window", "visual_focus",
                ],
            },
        },
        "good_plays": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "time": {"type": "string", "maxLength": 5},
                    "note": {"type": "string", "maxLength": 60},
                    "direct_answer": {"type": "string", "maxLength": 80},
                    "trigger": {"type": "string", "maxLength": 30},
                    "read": {"type": "string", "maxLength": 50},
                    "action": {"type": "string", "maxLength": 50},
                    "timing_window": {"type": "string", "maxLength": 40},
                    "reusable_rule": {"type": "string", "maxLength": 60},
                    "replay_window": {"type": "string", "maxLength": 15},
                    "visual_focus": {"type": "string", "maxLength": 40},
                },
                "required": [
                    "time", "note", "direct_answer", "trigger", "read",
                    "action", "timing_window", "reusable_rule",
                    "replay_window", "visual_focus",
                ],
            },
        },
    },
    "required": ["deaths", "good_plays"],
}


@dataclass(frozen=True)
class SceneCandidate:
    """Stage 1 内部で扱う候補シーン."""

    timestamp: str
    scene_type: str
    short_reason: str


def run_stage1_observer(video_path: str) -> Observation:
    """Stage 1: Observer — 実動画から観測結果を抽出する."""
    logger.info("Stage 1: Observer 実行")
    resolved_path = _validate_video_path(video_path)
    logger.info("video_path 検証成功: %s", resolved_path.name)
    client = _get_gemini_client()
    uploaded_file = _upload_video_file(client, resolved_path)
    try:
        active_file = _wait_for_uploaded_file(client, uploaded_file)
        overview_candidates = _run_stage1_overview_pass(client, active_file)
        detail_payload = _run_stage1_detail_pass(client, active_file, overview_candidates)
        structured_payload = _run_stage1_structured_pass(
            client,
            overview_candidates,
            detail_payload,
        )
        return _build_observation(structured_payload)
    finally:
        _delete_uploaded_file(client, uploaded_file)


def _validate_video_path(video_path: str) -> Path:
    """入力パスの妥当性を検証する."""
    normalized_path = video_path.strip()
    if not normalized_path:
        raise ValueError("video_path は空文字にできません")

    resolved_path = Path(normalized_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"動画ファイルが見つかりません: {normalized_path}")
    if not resolved_path.is_file():
        raise FileNotFoundError(f"動画ファイルではありません: {normalized_path}")
    if resolved_path.suffix.lower() != ".mp4":
        raise ValueError("動画ファイルは MP4 のみ対応です")
    return resolved_path


def _get_gemini_client() -> genai.Client:
    """Gemini API クライアントを返す."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY が未設定です")
    return genai.Client(api_key=api_key)


def _upload_video_file(client: genai.Client, video_path: Path) -> Any:
    """動画ファイルを File API へアップロードする."""
    size_mb = video_path.stat().st_size / (1024 * 1024)
    logger.info("File upload 開始: %s (%.1f MB)", video_path.name, size_mb)
    uploaded_file = client.files.upload(file=str(video_path), config={"mime_type": "video/mp4"})
    logger.info(
        "File upload 完了: name=%s, uri=%s",
        getattr(uploaded_file, "name", "?"),
        getattr(uploaded_file, "uri", "?"),
    )
    return uploaded_file


def _wait_for_uploaded_file(client: genai.Client, uploaded_file: Any) -> Any:
    """アップロード済みファイルが利用可能になるまで待つ."""
    file_name = getattr(uploaded_file, "name", None)
    if not file_name:
        raise RuntimeError("アップロード済みファイル名を取得できません")

    started_at = time.monotonic()
    while time.monotonic() - started_at < FILE_READY_TIMEOUT_SECONDS:
        current_file = client.files.get(name=file_name)
        state_name = _get_file_state_name(current_file)
        elapsed = time.monotonic() - started_at
        logger.info("File state: %s (%.0fs経過)", state_name, elapsed)
        if state_name == types.FileState.ACTIVE.value:
            return current_file
        if state_name == types.FileState.FAILED.value:
            error_detail = getattr(current_file, "error", None)
            mime_type = getattr(current_file, "mime_type", "unknown")
            size_bytes = getattr(current_file, "size_bytes", "unknown")
            raise RuntimeError(
                f"File API FAILED: error={error_detail}, mime_type={mime_type}, size_bytes={size_bytes}"
            )
        time.sleep(FILE_READY_POLL_INTERVAL_SECONDS)

    raise RuntimeError(
        f"File API タイムアウト ({FILE_READY_TIMEOUT_SECONDS}s): 最終state={_get_file_state_name(client.files.get(name=file_name))}"
    )


def _get_file_state_name(uploaded_file: Any) -> str:
    """アップロード済みファイルの state 名を文字列で返す."""
    state = getattr(uploaded_file, "state", None)
    if state is None:
        # state=None は STATE_UNSPECIFIED（処理中）として扱う（ACTIVE ではない）
        return "STATE_UNSPECIFIED"
    return getattr(state, "value", str(state))


def _run_stage1_overview_pass(client: genai.Client, uploaded_file: Any) -> list[SceneCandidate]:
    """Pass 1: 全体概要から候補シーンを抽出する."""
    logger.info("Pass 1 開始")
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[uploaded_file, PASS1_PROMPT],
        config=types.GenerateContentConfig(
            system_instruction=STAGE1_SYSTEM_PROMPT,
            response_mime_type="application/json",
        ),
    )
    payload = _load_json_payload(response, "Pass 1")
    logger.info("Pass 1 完了")
    return _build_candidates(payload)


def _build_candidates(payload: dict[str, object]) -> list[SceneCandidate]:
    """Pass 1 の JSON から候補一覧を構築する."""
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("Pass 1 の schema が不正です")

    candidates: list[SceneCandidate] = []
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, dict):
            raise ValueError("Pass 1 の schema が不正です")
        timestamp = _require_string(raw_candidate, "timestamp", stage_name="Pass 1")
        scene_type = _require_string(raw_candidate, "scene_type", stage_name="Pass 1")
        short_reason = _require_string(raw_candidate, "short_reason", stage_name="Pass 1")
        if scene_type not in {"death", "good_play"}:
            raise ValueError("Pass 1 の schema が不正です")
        if not TIME_PATTERN.fullmatch(timestamp):
            raise ValueError("Pass 1 の schema が不正です")
        candidates.append(SceneCandidate(timestamp, scene_type, short_reason))
    return candidates


def _run_stage1_detail_pass(
    client: genai.Client,
    uploaded_file: Any,
    candidates: list[SceneCandidate],
) -> dict[str, object]:
    """Pass 2: 候補シーンを clip 単位で詳細分析する."""
    logger.info("Pass 2 開始")
    parts = _build_detail_parts(uploaded_file, candidates)
    prompt = PASS2_PROMPT_TEMPLATE.format(candidate_text=_serialize_candidates(candidates))
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=types.Content(parts=[*parts, types.Part(text=prompt)]),
        config=types.GenerateContentConfig(
            system_instruction=STAGE1_SYSTEM_PROMPT,
            response_mime_type="application/json",
        ),
    )
    logger.info("Pass 2 完了")
    return _load_json_payload(response, "Pass 2")


def _build_detail_parts(uploaded_file: Any, candidates: list[SceneCandidate]) -> list[types.Part]:
    """Pass 2 用の動画 clip parts を構築する."""
    file_uri = getattr(uploaded_file, "uri", None)
    if not isinstance(file_uri, str) or not file_uri:
        raise RuntimeError("アップロード済み動画の file_uri を取得できません")

    duration_seconds = _extract_duration_seconds(uploaded_file)
    parts: list[types.Part] = []
    for candidate in candidates:
        start_second, end_second = _build_clip_window(candidate.timestamp, duration_seconds)
        parts.append(
            types.Part(
                file_data=types.FileData(file_uri=file_uri),
                video_metadata=types.VideoMetadata(
                    start_offset=f"{start_second}s",
                    end_offset=f"{end_second}s",
                ),
            )
        )
    return parts


def _build_clip_window(timestamp: str, duration_seconds: int | None) -> tuple[int, int]:
    """候補時刻から clip 窓を計算する."""
    center_second = _timestamp_to_seconds(timestamp)
    start_second = max(0, center_second - CLIP_WINDOW_SECONDS)
    end_second = center_second + CLIP_WINDOW_SECONDS
    if duration_seconds is not None:
        end_second = min(duration_seconds, end_second)
    return start_second, max(start_second + 1, end_second)


def _timestamp_to_seconds(timestamp: str) -> int:
    """MM:SS を秒に変換する."""
    minutes_str, seconds_str = timestamp.split(":")
    return int(minutes_str) * 60 + int(seconds_str)


def _extract_duration_seconds(uploaded_file: Any) -> int | None:
    """アップロード済みファイルから動画長を best-effort で取得する."""
    metadata = getattr(uploaded_file, "video_metadata", None)
    duration = getattr(metadata, "duration_seconds", None)
    if isinstance(duration, (int, float)):
        return int(duration)

    duration_millis = getattr(metadata, "duration_millis", None)
    if isinstance(duration_millis, (int, float)):
        return int(duration_millis / 1000)

    end_offset = getattr(metadata, "end_offset", None)
    if isinstance(end_offset, str) and end_offset.endswith("s"):
        try:
            return int(float(end_offset[:-1]))
        except ValueError:
            return None
    return None


def _serialize_candidates(candidates: list[SceneCandidate]) -> str:
    """候補一覧を JSON 文字列へ変換する."""
    return json.dumps(
        [
            {
                "timestamp": candidate.timestamp,
                "scene_type": candidate.scene_type,
                "short_reason": candidate.short_reason,
            }
            for candidate in candidates
        ],
        ensure_ascii=False,
    )


def _run_stage1_structured_pass(
    client: genai.Client,
    overview_candidates: list[SceneCandidate],
    detail_payload: dict[str, object],
) -> dict[str, object]:
    """Pass 3: 最終的な Observation JSON を確定する."""
    logger.info("Pass 3 開始")
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[
            PASS3_PROMPT_TEMPLATE.format(
                overview_text=_serialize_candidates(overview_candidates),
                detail_text=json.dumps(detail_payload, ensure_ascii=False),
            )
        ],
        config=types.GenerateContentConfig(
            system_instruction=STAGE1_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=OBSERVATION_SCHEMA,
        ),
    )
    logger.info("Pass 3 完了")
    return _load_json_payload(response, "Pass 3")


def _load_json_payload(response: Any, stage_name: str) -> dict[str, object]:
    """Gemini 応答から JSON を抽出する."""
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, dict):
        return parsed

    response_text = getattr(response, "text", None)
    if not isinstance(response_text, str) or not response_text.strip():
        raise ValueError(f"{stage_name} の応答が空です")

    try:
        return json.loads(response_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{stage_name} の JSON 解析に失敗しました") from error


def _build_observation(payload: dict[str, object]) -> Observation:
    """Pass 3 JSON を Observation へ変換する."""
    raw_deaths = payload.get("deaths")
    raw_good_plays = payload.get("good_plays")
    if not isinstance(raw_deaths, list) or not isinstance(raw_good_plays, list):
        logger.warning("Pass 3 JSON 検証失敗")
        raise ValueError("Pass 3 の schema が不正です")

    deaths = tuple(_build_death_scene(item) for item in raw_deaths)
    good_plays = tuple(_build_good_play(item) for item in raw_good_plays)
    return Observation(deaths=deaths, good_plays=good_plays)


def _build_death_scene(raw_item: object) -> DeathScene:
    """death 要素を DeathScene に変換する."""
    if not isinstance(raw_item, dict):
        logger.warning("Pass 3 JSON 検証失敗")
        raise ValueError("Pass 3 の schema が不正です")
    req = lambda field: _require_string(raw_item, field, stage_name="Pass 3")  # noqa: E731
    actor = raw_item.get("actor_champion", "")
    return DeathScene(
        time=_require_time(raw_item, stage_name="Pass 3"),
        cause=req("cause"),
        note=req("note"),
        root_cause=req("root_cause"),
        actor_champion=actor if isinstance(actor, str) else "",
        direct_answer=req("direct_answer"),
        improvement_answer=req("improvement_answer"),
        evidence=req("evidence"),
        counterfactual=req("counterfactual"),
        coach_rule=req("coach_rule"),
        replay_window=req("replay_window"),
        visual_focus=req("visual_focus"),
    )


def _build_good_play(raw_item: object) -> GoodPlay:
    """good_play 要素を GoodPlay に変換する."""
    if not isinstance(raw_item, dict):
        logger.warning("Pass 3 JSON 検証失敗")
        raise ValueError("Pass 3 の schema が不正です")
    req = lambda field: _require_string(raw_item, field, stage_name="Pass 3")  # noqa: E731
    return GoodPlay(
        time=_require_time(raw_item, stage_name="Pass 3"),
        note=req("note"),
        direct_answer=req("direct_answer"),
        trigger=req("trigger"),
        read=req("read"),
        action=req("action"),
        timing_window=req("timing_window"),
        reusable_rule=req("reusable_rule"),
        replay_window=req("replay_window"),
        visual_focus=req("visual_focus"),
    )


def _require_time(raw_item: dict[str, object], *, stage_name: str) -> str:
    """時刻文字列を取得して検証する."""
    time_text = _require_string(raw_item, "time", stage_name=stage_name)
    if not TIME_PATTERN.fullmatch(time_text):
        logger.warning("Pass 3 JSON 検証失敗")
        raise ValueError(f"{stage_name} の schema が不正です")
    return time_text


def _require_string(
    raw_item: dict[str, object],
    field_name: str,
    *,
    stage_name: str,
) -> str:
    """必須文字列フィールドを取得する."""
    value = raw_item.get(field_name)
    if not isinstance(value, str) or not value.strip():
        if stage_name == "Pass 3":
            logger.warning("Pass 3 JSON 検証失敗")
        raise ValueError(f"{stage_name} の schema が不正です")
    return value.strip()


def _delete_uploaded_file(client: genai.Client, uploaded_file: Any) -> None:
    """アップロード済みファイルを best-effort で削除する."""
    file_name = getattr(uploaded_file, "name", None)
    if not file_name:
        return
    try:
        client.files.delete(name=file_name)
    except Exception:
        logger.warning("File delete 失敗: %s", file_name, exc_info=True)


SCENE_DEATH_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "root_cause": {"type": "string", "enum": _ROOT_CAUSE_ENUM},
        "note": {"type": "string", "maxLength": 60},
        "direct_answer": {"type": "string", "maxLength": 80},
        "improvement_answer": {"type": "string", "maxLength": 80},
        "evidence": {"type": "string", "maxLength": 50},
        "counterfactual": {"type": "string", "maxLength": 50},
        "coach_rule": {"type": "string", "maxLength": 60},
        "visual_focus": {"type": "string", "maxLength": 40},
    },
    "required": [
        "root_cause", "note", "direct_answer", "improvement_answer",
        "evidence", "counterfactual", "coach_rule", "visual_focus",
    ],
}

SCENE_GOOD_PLAY_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "note": {"type": "string", "maxLength": 60},
        "direct_answer": {"type": "string", "maxLength": 80},
        "trigger": {"type": "string", "maxLength": 30},
        "read": {"type": "string", "maxLength": 50},
        "action": {"type": "string", "maxLength": 50},
        "timing_window": {"type": "string", "maxLength": 40},
        "reusable_rule": {"type": "string", "maxLength": 60},
        "visual_focus": {"type": "string", "maxLength": 40},
    },
    "required": [
        "note", "direct_answer", "trigger", "read",
        "action", "timing_window", "reusable_rule", "visual_focus",
    ],
}

SCENE_SYSTEM_PROMPT = """
あなたは League of Legends のコーチ向けアナリストです。
与えられた事実情報をもとに、動画のシーンを分析してください。
事実（チャンピオン名・時刻・キル情報）は変更しないでください。
あなたの役割は「なぜこれが起きたか」「どう改善するか」の分析だけです。

## 分析の原則

### 「観察」と「死因の推定」を分離せよ
- 動画で確認できた行動はすべて「観察事実」である
- 「死因」とは、その行動がなければ死ななかった直接的原因のことである
- 「〜していた時に死んだ」は相関であり、因果ではない
  - 悪い例: 「スコアボードを開いていたから死んだ」（スコアボード確認は正常な行動）
  - 良い例: 「視界のない川に単独で侵入したから死んだ」（立ち位置が直接原因）

### 相関と因果を混同するな
- 「デスの直前にしていた行動」＝「デスの原因」ではない
- 因果関係の判定基準: 「その行動を変えていれば、このデスは回避できたか？」
- この問いに YES と言えるものだけを root_cause / evidence に含めよ

### root_cause の選択基準
root_cause は以下の enum から1つ選択する:
  positioning_failure / information_failure / cooldown_disrespect /
  strength_misjudgment / wave_timing_failure / teamfight_execution /
  greedy_play / coordination_failure
「このデスの直接原因に最も合致するもの」を選べ。
「デスの直前にたまたま行っていた行動」に引きずられるな。

### evidence の基準
evidence には「死因に直接関係する観察事実のみ」を記載せよ。
動画で見えたすべてを列挙するのではなく、
「この事実がなければデスは発生しなかった」と言えるものだけを選べ。
""".strip()


def _analyze_single_scene(
    client: genai.Client,
    uploaded_file: Any,
    candidate: Any,
    scene_prompt: str,
    match_context: str,
) -> DeathScene | GoodPlay | None:
    """1 シーンを Gemini で分析し、DeathScene または GoodPlay を返す."""
    from sidecar.riot_api import SceneCandidate

    if not isinstance(candidate, SceneCandidate):
        return None

    file_uri = getattr(uploaded_file, "uri", None)
    if not isinstance(file_uri, str) or not file_uri:
        logger.warning("file_uri 取得失敗")
        return None

    schema = SCENE_DEATH_SCHEMA if candidate.scene_type == "death" else SCENE_GOOD_PLAY_SCHEMA
    system_instruction = f"{SCENE_SYSTEM_PROMPT}\n\n{match_context}"

    # 動画クリップ part
    clip_start_sec = candidate.clip_start_ms // 1000
    clip_end_sec = candidate.clip_end_ms // 1000
    video_part = types.Part(
        file_data=types.FileData(file_uri=file_uri),
        video_metadata=types.VideoMetadata(
            start_offset=f"{clip_start_sec}s",
            end_offset=f"{clip_end_sec}s",
        ),
    )

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=types.Content(parts=[video_part, types.Part(text=scene_prompt)]),
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        payload = _load_json_payload(response, f"Scene {candidate.scene_id}")
    except Exception:
        logger.warning("シーン分析失敗: %s", candidate.scene_id, exc_info=True)
        return None

    from sidecar.riot_api import ms_to_spoken
    time_str = ms_to_spoken(candidate.match_time_ms)
    replay_start = ms_to_spoken(candidate.clip_start_ms)
    replay_end = ms_to_spoken(candidate.clip_end_ms)
    replay_window = f"{replay_start}〜{replay_end}"

    if candidate.scene_type == "death":
        return DeathScene(
            time=time_str,
            cause=f"{candidate.actor_champion} による キル",
            note=str(payload.get("note", "")),
            root_cause=str(payload.get("root_cause", "positioning_failure")),
            actor_champion=candidate.actor_champion,
            direct_answer=str(payload.get("direct_answer", "")),
            improvement_answer=str(payload.get("improvement_answer", "")),
            evidence=str(payload.get("evidence", "")),
            counterfactual=str(payload.get("counterfactual", "")),
            coach_rule=str(payload.get("coach_rule", "")),
            replay_window=replay_window,
            visual_focus=str(payload.get("visual_focus", "")),
        )

    return GoodPlay(
        time=time_str,
        note=str(payload.get("note", "")),
        direct_answer=str(payload.get("direct_answer", "")),
        trigger=str(payload.get("trigger", "")),
        read=str(payload.get("read", "")),
        action=str(payload.get("action", "")),
        timing_window=str(payload.get("timing_window", "")),
        reusable_rule=str(payload.get("reusable_rule", "")),
        replay_window=replay_window,
        visual_focus=str(payload.get("visual_focus", "")),
    )


def run_stage3_analyst(observation: Observation, raw: RawMatchData | None = None) -> Analysis:
    """Stage 3: Analyst — パターンを抽出する."""
    logger.info("Stage 3: Analyst 実行")

    total_deaths = len(observation.deaths)

    # root_cause ベースの弱みパターン検出
    if total_deaths > 0:
        root_causes = [d.root_cause for d in observation.deaths]
        main_root_cause = max(set(root_causes), key=root_causes.count)
        root_cause_count = root_causes.count(main_root_cause)
        root_label = _ROOT_CAUSE_LABELS.get(main_root_cause, main_root_cause)
        weakness = f"{root_label}（{root_cause_count}/{total_deaths}デス）"
        main_weakness_cause = main_root_cause
    else:
        weakness = "特になし"
        main_weakness_cause = ""

    # 強み: good_play があればそこから、なければデフォルト
    if observation.good_plays:
        strength = f"良プレイ {len(observation.good_plays)} 件検出"
    else:
        strength = "分析データ不足"

    return Analysis(
        strength=strength,
        weakness=weakness,
        good_scenes=tuple(play.time for play in observation.good_plays),
        bad_scenes=tuple(death.time for death in observation.deaths[:2]),
        good_count=len(observation.good_plays),
        total_deaths=total_deaths,
        main_weakness_cause=main_weakness_cause,
    )


# ---------------------------------------------------------------------------
# ヘルパー: フェーズ・ランキング・デス分類
# ---------------------------------------------------------------------------

_EARLY_MS = 14 * 60_000
_MID_MS = 25 * 60_000


def _phase_of(ms: int) -> str:
    """ミリ秒からフェーズキーを返す."""
    if ms < _EARLY_MS:
        return "early"
    if ms < _MID_MS:
        return "mid"
    return "late"


def _compute_phase_kda(raw: RawMatchData) -> dict[str, tuple[int, int, int]]:
    """フェーズ別 (kills, deaths, assists) を計算する."""
    player = raw.meta.player_champion
    if not player:
        return {}
    buckets: dict[str, list[int]] = {
        "early": [0, 0, 0], "mid": [0, 0, 0], "late": [0, 0, 0],
    }
    for ev in raw.kill_events:
        phase = _phase_of(ev.timestamp_ms)
        if ev.killer_champion == player:
            buckets[phase][0] += 1
        if ev.victim_champion == player:
            buckets[phase][1] += 1
        if player in ev.assist_champions:
            buckets[phase][2] += 1
    return {k: (v[0], v[1], v[2]) for k, v in buckets.items()}


def _ally_rank(raw: RawMatchData, stat_key: str) -> tuple[int, int]:
    """味方チーム内の順位を返す (rank, total). 見つからなければ (0, 0)."""
    player = raw.meta.player_champion
    allies = [p for p in raw.participants if p.team == "ally"]
    if not allies or not player:
        return (0, 0)
    values = sorted(
        [(getattr(p, stat_key, 0), p.champion) for p in allies],
        key=lambda x: x[0], reverse=True,
    )
    for i, (_, champ) in enumerate(values):
        if champ == player:
            return (i + 1, len(values))
    return (0, 0)


def _classify_death_severity(death_ms: int, raw: RawMatchData) -> str:
    """デスの深刻度を分類する."""
    nearby_kills = [
        k for k in raw.kill_events
        if abs(k.timestamp_ms - death_ms) < 15_000
        and k.timestamp_ms != death_ms
    ]
    is_teamfight = len(nearby_kills) >= 2
    team_gained = any(
        o for o in raw.objective_events
        if 0 < o.timestamp_ms - death_ms <= 60_000 and o.killer_team == "ally"
    )
    enemy_gained = any(
        o for o in raw.objective_events
        if 0 < o.timestamp_ms - death_ms <= 60_000 and o.killer_team == "enemy"
    )
    enemy_tower = any(
        b for b in raw.building_events
        if 0 < b.timestamp_ms - death_ms <= 60_000 and b.lost_team == "ally"
    )
    if not is_teamfight and (enemy_gained or enemy_tower):
        return "致命的ミス"
    if is_teamfight and team_gained:
        return "集団戦（トレード）"
    if is_teamfight:
        return "集団戦デス"
    if team_gained:
        return "トレード"
    return "判断ミス"


_ROOT_CAUSE_LABELS: dict[str, str] = {
    "positioning_failure": "立ち位置ミス",
    "information_failure": "情報不足",
    "cooldown_disrespect": "CD 管理ミス",
    "strength_misjudgment": "強さ誤判断",
    "wave_timing_failure": "ウェーブ判断ミス",
    "teamfight_execution": "集団戦実行ミス",
    "greedy_play": "欲張りムーブ",
    "coordination_failure": "連携ミス",
}

_SYSTEM_INSTRUCTION_TEMPLATE = """\
[1] Role & Identity
あなたは RePlection — League of Legends の振り返りを一緒にしてくれる友達です。
- ポジティブで正直。課題はダメ出しではなく「発見」として伝える
- 「見せて」「止めて」にはリプレイ操作で必ず対応する

[1.5] First Turn（最初の発話）
会話が始まったら、最初のターンで [2]〜[6] の実データを使って以下を一気に話す:
1. 挨拶と試合結果（例:「お疲れ！勝ったね！」※[2]の勝敗を使う）
2. KDA の感想（例:「アシスト多いじゃん」※[2]のKDAを使う）
3. [3] Topline Positives から良かった点を 1-2 個
4. [4] Habit Patterns から癖の発見
5. [5] Scene Cards の最初のデスを 1 つ簡潔に
6. [6] Next-game Focus から次の 1 つ
7. 「気になるところある？」で相手に渡す

重要: 必ず [2]〜[6] の実データを参照して話す。ハードコードの数値を使わない。
これは最初の 1 ターンだけ。その後の会話では 1〜2 文で短く返す。
時間を言うときは「〇〇分〇〇秒」形式。「〇〇時」とは絶対に言わない。

[2] Match Snapshot
{match_snapshot}

[3] Topline Positives
{topline_positives}

[4] Habit Patterns
{habit_patterns}

[5] 試合展開（タイムライン）
{match_timeline}

[6] Scene Cards
--- Deaths ---
{death_cards}

--- Good Plays ---
{good_play_cards}

[7] Next-game Focus
{next_game_focus}

[8] Conversation Policy
「どうだった？」             → まず良かった点から入る → 次に癖の発見
「なぜ死んだ？」             → direct_answer → replay_window でシーンを見せる
「何を直す？」               → improvement_answer → coach_rule を添える
「何が良かった？」           → direct_answer → reusable_rule を添える
「次の試合で 1 つ意識するなら」→ [7] Next-game Focus を返す
「〜は避けられた？」         → counterfactual を答える
「バロンどうだった？」       → [5] 試合展開のオブジェクト情報を使って答える
「ドラゴンは？」             → [5] 試合展開のオブジェクト情報を使って答える

[10] Replay Tool Policy
- シーンを見せるときは seek_replay(timestamp) で飛ばし、再生を続けながら説明する
- 自分から pause_replay() を呼ばない。ユーザーが「止めて」と言った時だけ pause する
- 「再開」→ resume_replay()
- 「スロー」→ slow_motion(speed=0.5)
- 時間を言うときは「〇〇分〇〇秒」形式で。「〇〇時」とは絶対に言わない

[11] Uncertainty
- 動画で確認できなかった情報は推測と明示する
- 画面外の行動は「おそらく〜」と断る"""


_MAX_DEATH_CARDS = 4
_MAX_GOOD_PLAY_CARDS = 2


def _format_death_cards(deaths: tuple[DeathScene, ...]) -> str:
    """Death scene cards を system_instruction 用テキストに変換する."""
    if not deaths:
        return "（記録されたデスシーンはありません）"
    cards = []
    for d in deaths[:_MAX_DEATH_CARDS]:
        root_label = _ROOT_CAUSE_LABELS.get(d.root_cause, d.root_cause)
        cards.append(
            f"[死亡 {d.time}] {d.cause}（{root_label}）\n"
            f"  Q「なぜ死んだ？」    → {d.direct_answer}\n"
            f"  Q「何を直す？」      → {d.improvement_answer}\n"
            f"  根拠: {d.evidence}\n"
            f"  代替行動: {d.counterfactual}\n"
            f"  ルール: {d.coach_rule}\n"
            f"  Replay: {d.replay_window} ／ 注目: {d.visual_focus}"
        )
    return "\n\n".join(cards)


def _format_good_play_cards(good_plays: tuple[GoodPlay, ...]) -> str:
    """Good play cards を system_instruction 用テキストに変換する."""
    if not good_plays:
        return "（記録された良プレイはありません）"
    cards = []
    for g in good_plays[:_MAX_GOOD_PLAY_CARDS]:
        cards.append(
            f"[良プレイ {g.time}] {g.note}\n"
            f"  Q「何が良かった？」  → {g.direct_answer}\n"
            f"  起点: {g.trigger} ／ 認識: {g.read}\n"
            f"  行動: {g.action} ／ タイミング: {g.timing_window}\n"
            f"  ルール: {g.reusable_rule}\n"
            f"  Replay: {g.replay_window} ／ 注目: {g.visual_focus}"
        )
    return "\n\n".join(cards)


_DEFAULT_RULES: dict[str, str] = {
    "positioning_failure": "集団戦や移動時に、味方から離れすぎない位置を保つ",
    "information_failure": "視界のない場所に入る前に、ミニマップで敵の位置を確認する",
    "strength_misjudgment": "戦う前に相手のレベルとアイテムを確認してから仕掛ける",
    "cooldown_disrespect": "重要スキルのクールダウンを意識してから戦闘に入る",
    "execution_error": "スキルの順番とタイミングを意識して丁寧に操作する",
    "coordination_failure": "味方のスキルと足並みを揃えてから仕掛ける",
}


def _get_valid_rule(deaths: tuple[DeathScene, ...], root_cause: str) -> str:
    """有効な coach_rule を取得する。Gemini の分析エラーテキストが混入しないようフィルタ."""
    for d in deaths:
        if d.root_cause == root_cause and d.coach_rule:
            rule = d.coach_rule
            # 分析エラーっぽいテキストを除外
            if any(bad in rule for bad in ["特定できません", "映像", "分析", "推測", "スコアボード確認は"]):
                continue
            return rule
    return _DEFAULT_RULES.get(root_cause, "安全な立ち位置を意識する")


def _select_next_game_focus(
    deaths: tuple[DeathScene, ...],
    observation: Observation | None = None,
    raw: RawMatchData | None = None,
) -> str:
    """次の試合で意識することを3段構造で生成する."""
    lines: list[str] = []

    # 最重要: 最頻出の root_cause
    if deaths:
        causes = [d.root_cause for d in deaths]
        most_common = max(set(causes), key=causes.count)
        count = causes.count(most_common)
        label = _ROOT_CAUSE_LABELS.get(most_common, most_common)
        rule = _get_valid_rule(deaths, most_common)
        examples = [d.time for d in deaths if d.root_cause == most_common]
        lines.append(f"最重要: {rule}")
        lines.append(f"  根拠: {label}が{count}回（{', '.join(examples[:3])}）")

        # 次点: 2番目の root_cause があれば
        cause_counts = Counter(causes)
        if len(cause_counts) >= 2:
            second_cause, second_count = cause_counts.most_common(2)[1]
            if second_count >= 2:
                second_label = _ROOT_CAUSE_LABELS.get(second_cause, second_cause)
                second_rule = _get_valid_rule(deaths, second_cause)
                lines.append(f"\n次点: {second_rule}")
                lines.append(f"  根拠: {second_label}が{second_count}回")
    else:
        lines.append("特になし — デスなしの試合")

    # 継続: 良プレイがあれば褒めて維持を促す
    if observation and observation.good_plays:
        gp = observation.good_plays[0]
        lines.append(f"\n継続: {gp.reusable_rule}")
        lines.append(f"  根拠: {gp.time}のプレイで実践済み。この意識を維持")

    return "\n".join(lines)


def _build_match_timeline(raw: RawMatchData | None) -> str:
    """試合展開タイムラインを生成する（オブジェクト・タワー・主要キル）."""
    if raw is None:
        return "試合データなし"

    from sidecar.riot_api import ms_to_spoken
    lines: list[str] = []

    # 全イベントを時系列でまとめる
    events: list[tuple[int, str]] = []

    # オブジェクト
    for o in raw.objective_events:
        team = "味方" if o.killer_team == "ally" else "敵"
        events.append((o.timestamp_ms, f"{team}が{o.monster_type}獲得"))

    # タワー・インヒビター
    for b in raw.building_events:
        lost = "敵" if b.lost_team == "enemy" else "味方"
        btype = "タワー" if "TOWER" in b.building_type else "インヒビター"
        events.append((b.timestamp_ms, f"{lost}の{btype}破壊"))

    # 自分のデス（主要イベントとして）
    player = raw.meta.player_champion
    for k in raw.kill_events:
        if k.victim_champion == player:
            events.append((k.timestamp_ms, f"{player}が{k.killer_champion}にキルされた"))
        elif k.killer_champion == player:
            events.append((k.timestamp_ms, f"{player}が{k.victim_champion}をキル"))

    # 時系列ソート
    events.sort(key=lambda x: x[0])

    # フェーズ別に整理
    early = [(t, e) for t, e in events if t < 14 * 60_000]
    mid = [(t, e) for t, e in events if 14 * 60_000 <= t < 25 * 60_000]
    late = [(t, e) for t, e in events if t >= 25 * 60_000]

    for phase_name, phase_events in [("序盤(0-14分)", early), ("中盤(14-25分)", mid), ("終盤(25分-)", late)]:
        if phase_events:
            lines.append(f"【{phase_name}】")
            for ts, desc in phase_events:
                lines.append(f"  {ms_to_spoken(ts)}: {desc}")

    return "\n".join(lines)


def run_stage4_coach(
    analysis: Analysis,
    observation: Observation,
    raw: RawMatchData | None = None,
) -> str:
    """Stage 4: Coach — Gemini Live API 用 system_instruction を生成する."""
    logger.info("Stage 4: Coach 実行")
    return _SYSTEM_INSTRUCTION_TEMPLATE.format(
        match_snapshot=_build_match_snapshot(raw),
        topline_positives=_build_topline_positives(observation, raw),
        habit_patterns=_build_habit_patterns(observation, raw),
        match_timeline=_build_match_timeline(raw),
        death_cards=_format_death_cards(observation.deaths),
        good_play_cards=_format_good_play_cards(observation.good_plays),
        next_game_focus=_select_next_game_focus(observation.deaths, observation, raw),
    )


def _snapshot_team_rankings(raw: RawMatchData) -> list[str]:
    """チーム内ランキングで1位の項目をまとめて返す."""
    rankings: list[str] = []
    for key, label in [("total_damage_dealt", "ダメージ"), ("vision_score", "ビジョン"), ("assists", "アシスト")]:
        rank, total = _ally_rank(raw, key)
        if rank == 1 and total > 1:
            rankings.append(f"{label}1位")
    if rankings:
        return [f"チーム貢献: {' / '.join(rankings)}"]
    return []


def _snapshot_phase_kda(raw: RawMatchData) -> list[str]:
    """フェーズ別KDAの表示行を返す."""
    phase_kda = _compute_phase_kda(raw)
    if not phase_kda or not any(sum(v) > 0 for v in phase_kda.values()):
        return []
    lines = ["\n【フェーズ別】"]
    for pk, label in [("early", "序盤(0-14分)"), ("mid", "中盤(14-25分)"), ("late", "終盤(25分-)")]:
        k, d, a = phase_kda.get(pk, (0, 0, 0))
        if k or d or a:
            lines.append(f"{label}: {k}/{d}/{a}")
    return lines


def _build_match_snapshot(raw: RawMatchData | None) -> str:
    """試合サマリーを生成する（enriched: フェーズ別KDA・チーム貢献・デス分類）."""
    if raw is None:
        return "試合データなし"

    meta = raw.meta
    duration_min = meta.game_duration_ms // 60_000
    duration_sec = (meta.game_duration_ms % 60_000) // 1000
    result_str = "勝利" if meta.win else "敗北"

    my = _find_player_stats(raw)
    opp = _find_opponent_stats(raw)

    lines = [f"【結果】{result_str} / {duration_min}:{duration_sec:02d} / {meta.player_champion}({meta.player_role})"]

    if my:
        lines.append(f"KDA: {my.kills}/{my.deaths}/{my.assists}")
        lines.append(f"CS: {my.total_cs} / ダメージ: {my.total_damage_dealt:,} / ビジョン: {my.vision_score}")

    if my and opp:
        cs_d = my.total_cs - opp.total_cs
        dmg_d = my.total_damage_dealt - opp.total_damage_dealt
        lines.append(f"対面({opp.champion}): CS{'+' if cs_d >= 0 else ''}{cs_d} / ダメージ{'+' if dmg_d >= 0 else ''}{dmg_d:,}")

    lines.extend(_snapshot_team_rankings(raw))
    lines.extend(_snapshot_objectives(raw))
    lines.extend(_snapshot_win_cause(raw, my))
    lines.extend(_snapshot_phase_kda(raw))
    lines.extend(_snapshot_death_classification(raw))

    return "\n".join(lines)


def _find_player_stats(raw: RawMatchData) -> ParticipantStats | None:
    """プレイヤー自身の stats を返す."""
    for p in raw.participants:
        if p.champion == raw.meta.player_champion:
            return p
    return None


def _find_opponent_stats(raw: RawMatchData) -> ParticipantStats | None:
    """対面の stats を返す."""
    for p in raw.participants:
        if p.role == raw.meta.player_role and p.team == "enemy":
            return p
    return None


def _count_objectives(raw: RawMatchData, team: str, monster_substr: str) -> int:
    """指定チーム・モンスター種別のオブジェクト獲得数を返す."""
    return sum(
        1 for o in raw.objective_events
        if o.killer_team == team and monster_substr in o.monster_type
    )


def _snapshot_objectives(raw: RawMatchData) -> list[str]:
    """オブジェクトサマリー行を返す."""
    lines: list[str] = []
    ally_dr = _count_objectives(raw, "ally", "DRAGON")
    enemy_dr = _count_objectives(raw, "enemy", "DRAGON")
    ally_baron = _count_objectives(raw, "ally", "BARON")
    ally_herald = _count_objectives(raw, "ally", "RIFTHERALD")

    parts: list[str] = []
    if ally_dr or enemy_dr:
        note = f"ドラゴン{ally_dr}本"
        if ally_dr >= 4:
            note += "（ソウル達成）"
        parts.append(note)
    if ally_baron:
        parts.append(f"バロン{ally_baron}回")
    if ally_herald:
        parts.append(f"ヘラルド{ally_herald}回")
    if parts:
        lines.append(f"オブジェクト: {' / '.join(parts)}")

    ally_tw = sum(1 for b in raw.building_events if b.lost_team == "enemy" and "TOWER" in b.building_type)
    enemy_tw = sum(1 for b in raw.building_events if b.lost_team == "ally" and "TOWER" in b.building_type)
    if ally_tw or enemy_tw:
        lines.append(f"タワー: 破壊{ally_tw} / 被破壊{enemy_tw}")
    return lines


def _snapshot_win_cause(raw: RawMatchData, my: ParticipantStats | None) -> list[str]:
    """勝因/敗因分析行を返す."""
    lines: list[str] = []
    ally_dr = _count_objectives(raw, "ally", "DRAGON")
    ally_baron = _count_objectives(raw, "ally", "BARON")

    if raw.meta.win:
        causes: list[str] = []
        if ally_dr >= 4:
            causes.append("ドラゴンソウル達成")
        if ally_baron >= 1:
            causes.append("バロン活用")
        if my and my.assists >= 10:
            causes.append(f"高アシスト({my.assists})")
        if causes:
            lines.append(f"勝因: {' + '.join(causes)}")
    else:
        causes = []
        if my and my.deaths >= 6:
            causes.append(f"デス多い({my.deaths})")
        if causes:
            lines.append(f"敗因候補: {' + '.join(causes)}")
    return lines


def _snapshot_death_classification(raw: RawMatchData) -> list[str]:
    """デス分類サマリー行を返す."""
    player_deaths = [
        ev for ev in raw.kill_events if ev.victim_champion == raw.meta.player_champion
    ]
    if not player_deaths:
        return []
    severities = [_classify_death_severity(ev.timestamp_ms, raw) for ev in player_deaths]
    sev_counts = Counter(severities)
    sev_parts = [f"{sev}{cnt}回" for sev, cnt in sev_counts.most_common()]
    lines = [f"\n【デス分類】{len(player_deaths)}デス: {' / '.join(sev_parts)}"]
    solo = sum(1 for ev in player_deaths if not ev.assist_champions)
    if solo:
        lines.append(f"単独デス: {solo}回")
    return lines


def _build_topline_positives(observation: Observation, raw: RawMatchData | None) -> str:
    """良かった点を 2-3 個、根拠付きで生成する."""
    positives: list[str] = []

    # 良プレイから（最大2件）
    for gp in observation.good_plays[:2]:
        positives.append(f"{gp.time}: {gp.direct_answer}")

    if raw and raw.meta.player_champion:
        my = _find_player_stats(raw)
        if my:
            # KDA が良ければ
            if my.kills + my.assists >= my.deaths * 2 and my.deaths > 0:
                positives.append(f"KDA {my.kills}/{my.deaths}/{my.assists} は貢献度が高い")
            elif my.deaths == 0:
                positives.append("デスなし — 完璧な生存")

            # ビジョンがチーム1位
            vr, _ = _ally_rank(raw, "vision_score")
            if vr == 1 and my.vision_score >= 20:
                positives.append(f"ビジョンスコア{my.vision_score}（チーム最高） — マップ管理に貢献")

            # アシストがチーム1位
            ar, _ = _ally_rank(raw, "assists")
            if ar == 1 and my.assists >= 8:
                positives.append(f"アシスト{my.assists}（チーム最多） — チームファイトへの参加率が高い")

            # ダメージ効率（CS低いのにダメージ高い）
            opp = _find_opponent_stats(raw)
            if opp and my.total_cs < opp.total_cs and my.total_damage_dealt > opp.total_damage_dealt:
                positives.append("対面よりCS負けてるがダメージは勝っている — 効率的な戦闘")

            # ドラゴンソウル
            ally_dr = _count_objectives(raw, "ally", "DRAGON")
            if ally_dr >= 4:
                positives.append(f"ドラゴン{ally_dr}本でソウル達成 — オブジェクトコントロール")

    if not positives:
        positives.append("分析データから良い点を探し中")

    return "\n".join(f"- {p}" for p in positives[:3])


def _build_habit_patterns(observation: Observation, raw: RawMatchData | None = None) -> str:
    """繰り返す癖を検出する（デス分類・フェーズ集中を含む）."""
    if not observation.deaths:
        return "（デスなし — 癖の検出対象なし）"

    patterns: list[str] = []

    # root_cause パターン
    root_counts = Counter(d.root_cause for d in observation.deaths)
    for cause, count in root_counts.most_common(2):
        if count >= 2:
            label = _ROOT_CAUSE_LABELS.get(cause, cause)
            examples = [d.time for d in observation.deaths if d.root_cause == cause]
            patterns.append(f"{label} が {count} 回（{', '.join(examples[:3])}）")

    # デス分類パターン（raw がある場合）
    if raw:
        player_deaths = [
            ev for ev in raw.kill_events if ev.victim_champion == raw.meta.player_champion
        ]
        if player_deaths:
            severities = [_classify_death_severity(ev.timestamp_ms, raw) for ev in player_deaths]
            sev_counts = Counter(severities)
            critical = sev_counts.get("致命的ミス", 0) + sev_counts.get("判断ミス", 0)
            if critical >= 2:
                patterns.append(f"コーチすべきデスは{critical}回（致命的ミス+判断ミス）。残りは集団戦やトレード")

            # フェーズ集中パターン
            phase_deaths = Counter(_phase_of(ev.timestamp_ms) for ev in player_deaths)
            worst_phase, worst_count = phase_deaths.most_common(1)[0]
            if worst_count >= 3:
                label = {"early": "序盤", "mid": "中盤", "late": "終盤"}.get(worst_phase, worst_phase)
                patterns.append(f"{label}にデスが集中（{worst_count}回）")

            # 単独デスパターン
            solo = sum(1 for ev in player_deaths if not ev.assist_champions)
            if solo >= 3:
                patterns.append(f"単独デスが{solo}回 — 孤立する場面が多い")

    if not patterns:
        patterns.append("明確な繰り返しパターンなし")

    return "\n".join(f"- {p}" for p in patterns)


def _cache_key(video_path: str, riot_fingerprint: str = "") -> str:
    """sha256(abs_path)[:16] + mtime_ns + size + riot_fingerprint でキャッシュキーを生成する."""
    abs_path = os.path.abspath(video_path)
    stat = os.stat(abs_path)
    h = hashlib.sha256(abs_path.encode()).hexdigest()[:16]
    base = f"{h}_{stat.st_mtime_ns}_{stat.st_size}"
    if riot_fingerprint:
        return f"{base}_{riot_fingerprint}"
    return base


def load_cached_context(video_path: str, riot_fingerprint: str = "") -> CoachingContext | None:
    """キャッシュヒット時に CoachingContext を返す。検証失敗時は None（再分析）."""
    key = _cache_key(video_path, riot_fingerprint)
    cache_file = SESSIONS_DIR / f"{key}.json"
    if not cache_file.exists():
        return None
    try:
        return CoachingContext.from_json(cache_file.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("キャッシュ読み込み失敗（再分析）: %s", cache_file)
        return None


def save_context(ctx: CoachingContext, video_path: str, riot_fingerprint: str = "") -> Path:
    """CoachingContext を data/sessions/{cache_key}.json に保存して Path を返す."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(video_path, riot_fingerprint)
    cache_file = SESSIONS_DIR / f"{key}.json"
    cache_file.write_text(ctx.to_json(), encoding="utf-8")
    logger.info("コンテキスト保存: %s", cache_file)
    return cache_file


def context_to_system_instruction(ctx: CoachingContext) -> str:
    """CoachingContext から Gemini Live API 用 system_instruction を生成する."""
    return ctx.system_instruction


def run_analyze(video_path: str, match_id: str = "") -> tuple[CoachingContext, Path]:
    """Stage A→B→C パイプラインを実行し、(CoachingContext, 保存パス) を返す.

    Stage A: Riot API で事実データ取得 + シーン候補生成
    Stage B: Gemini で各シーンの「なぜ」を動画分析
    Stage C: 統合して CoachingContext を生成

    キャッシュが有効な場合はスキップして即座に返す。
    """
    from sidecar.riot_api import (
        build_match_context_prompt,
        build_scene_analysis_prompt,
        cache_fingerprint,
        fetch_match_data,
    )

    # --- Stage A: Riot Grounding ---
    logger.info("Stage A: Riot Grounding 開始")
    match_info, candidates, raw_match_data = fetch_match_data(match_id)
    puuid = os.getenv("RIOT_PUUID", "").strip() or None
    fingerprint = cache_fingerprint(match_info.match_id, puuid)

    # キャッシュチェック（Riot API の後、Gemini API の前）
    cached = load_cached_context(video_path, fingerprint)
    if cached is not None:
        logger.info("キャッシュヒット: %s", video_path)
        cache_path = SESSIONS_DIR / f"{_cache_key(video_path, fingerprint)}.json"
        return cached, cache_path

    logger.info("Stage A 完了: %d シーン候補", len(candidates))

    # --- Stage B: Vision Analysis ---
    logger.info("Stage B: Vision Analysis 開始")
    resolved_path = _validate_video_path(video_path)
    client = _get_gemini_client()
    uploaded_file = _upload_video_file(client, resolved_path)

    try:
        active_file = _wait_for_uploaded_file(client, uploaded_file)
        match_context = build_match_context_prompt(match_info)

        death_scenes: list[DeathScene] = []
        good_plays: list[GoodPlay] = []

        for candidate in candidates:
            logger.info("シーン分析: %s (%s)", candidate.scene_id, candidate.scene_type)
            scene_prompt = build_scene_analysis_prompt(candidate, match_info, raw_match_data)
            scene_result = _analyze_single_scene(
                client, active_file, candidate, scene_prompt, match_context,
            )
            if candidate.scene_type == "death" and scene_result is not None:
                death_scenes.append(scene_result)
            elif candidate.scene_type == "good_play" and scene_result is not None:
                good_plays.append(scene_result)

    finally:
        _delete_uploaded_file(client, uploaded_file)

    observation = Observation(
        deaths=tuple(death_scenes),
        good_plays=tuple(good_plays),
    )
    logger.info(
        "Stage B 完了: deaths=%d, good_plays=%d",
        len(observation.deaths),
        len(observation.good_plays),
    )

    # --- Stage C: Synthesis ---
    logger.info("Stage C: Synthesis 開始")
    video_source = Path(video_path).name

    analysis = run_stage3_analyst(observation, raw_match_data)
    system_instruction = run_stage4_coach(analysis, observation, raw_match_data)
    logger.info("Stage C 完了: system_instruction %d 文字", len(system_instruction))

    ctx = CoachingContext.from_observation_analysis(
        video_source=video_source,
        match_info=match_info,
        obs=observation,
        analysis=analysis,
        system_instruction=system_instruction,
        raw_match_data=raw_match_data,
    )

    # キャッシュ保存（fingerprint 付き）
    saved_path = save_context(ctx, video_path, fingerprint)
    return ctx, saved_path
