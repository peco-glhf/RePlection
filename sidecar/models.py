"""データモデル定義.

sidecar 全体で使用するデータクラスと Enum を定義する。
すべての dataclass は frozen=True（不変）で定義し、安全な値の受け渡しを保証する。
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum


class CoachState(str, Enum):
    """コーチングセッション全体の状態."""

    IDLE = "idle"
    ANALYZING = "analyzing"  # 動画分析中（POST /analyze）
    READY = "ready"          # 分析完了・Live 起動待ち
    DIALOGUE = "dialogue"    # Live API 対話中
    ERROR = "error"


class DialogueState(str, Enum):
    """Gemini Live API 対話の状態（割り込み制御に使用）."""

    IDLE = "idle"
    LISTENING = "listening"  # ユーザー発話待ち
    USER_SPEAKING = "user_speaking"  # Push-to-talk でユーザー発話中
    PROCESSING = "processing"  # Gemini 処理中
    SPEAKING = "speaking"  # AI 発話中
    INTERRUPTED = "interrupted"  # barge-in 発生


@dataclass(frozen=True)
class DeathScene:
    """1 件のデスシーン情報."""

    time: str              # 例: "15:41"
    cause: str             # 即時死因 例: "マルファイト ULT"
    note: str              # 観察メモ 例: "ULT 射程内に立ち続けた"
    root_cause: str        # 根本原因分類 例: "positioning_failure"
    actor_champion: str    # Riot API 由来の正規化チャンピオン名。非チャンプ死因は空文字列
    direct_answer: str     # 「なぜ死んだ？」への 1 文直接回答
    improvement_answer: str  # 「何を直すべき？」への 1 文回答
    evidence: str          # 死亡直前に観測できた具体的事実
    counterfactual: str    # 回避できた代替行動
    coach_rule: str        # 次の試合で使える再利用可能なルール文
    replay_window: str     # 推奨再生範囲 例: "15:38-15:47"
    visual_focus: str      # 視聴者が注目すべき画面上の要素


@dataclass(frozen=True)
class GoodPlay:
    """1 件の良プレイ情報."""

    time: str           # 例: "08:39"
    note: str           # プレイの 1 行説明
    direct_answer: str  # 「何が良かった？」への 1 文回答
    trigger: str        # このプレイを起こした起点イベント
    read: str           # 認識したシグナル
    action: str         # 実行した行動の順序
    timing_window: str  # 行動が有効だった時間的条件
    reusable_rule: str  # 同じ状況で再現できるルール
    replay_window: str  # 推奨再生範囲 例: "08:36-08:44"
    visual_focus: str   # 視聴者が注目すべき画面上の要素


@dataclass(frozen=True)
class Observation:
    """Stage 1（Observer）の出力.

    タプルで保持することで不変性を保証する。
    """

    deaths: tuple[DeathScene, ...]
    good_plays: tuple[GoodPlay, ...]


@dataclass(frozen=True)
class Analysis:
    """Stage 3（Analyst）の出力."""

    strength: str  # 例: "予測力（反応速度でなく先読み）"
    weakness: str  # 例: "エンゲージスキルへの被弾（6/10デス）"
    good_scenes: tuple[str, ...]  # 代表タイムスタンプ（良プレイ）
    bad_scenes: tuple[str, ...]  # 代表タイムスタンプ（弱みシーン）
    good_count: int
    total_deaths: int
    main_weakness_cause: str  # 例: "マルファイト ULT"


@dataclass(frozen=True)
class Participant:
    """試合参加者の基本情報（Riot API 由来）."""

    champion: str   # 例: "Zed"
    role: str       # TOP / JUNGLE / MIDDLE / BOTTOM / UTILITY
    team: str       # "ally" / "enemy" / "unknown"


@dataclass(frozen=True)
class ParticipantStats:
    """試合参加者の詳細統計（Riot API 由来）."""

    champion: str
    role: str
    team: str  # "ally" / "enemy" / "unknown"
    kills: int
    deaths: int
    assists: int
    total_damage_dealt: int
    total_damage_taken: int
    total_cs: int  # minions + jungle
    gold_earned: int
    vision_score: int
    wards_placed: int
    wards_killed: int


@dataclass(frozen=True)
class KillEvent:
    """キルイベント（Timeline 由来）."""

    timestamp_ms: int
    killer_champion: str
    victim_champion: str
    assist_champions: tuple[str, ...]
    position_x: int
    position_y: int
    bounty: int


@dataclass(frozen=True)
class ObjectiveEvent:
    """オブジェクトイベント（ドラゴン/バロン/ヘラルド）."""

    timestamp_ms: int
    monster_type: str  # DRAGON / BARON_NASHOR / RIFTHERALD
    killer_team: str   # "ally" / "enemy" / "unknown"


@dataclass(frozen=True)
class BuildingEvent:
    """建物破壊イベント."""

    timestamp_ms: int
    building_type: str  # TOWER_BUILDING / INHIBITOR_BUILDING
    lost_team: str      # ally / enemy（破壊された側）


@dataclass(frozen=True)
class WardEvent:
    """ワードイベント."""

    timestamp_ms: int
    event_type: str      # "placed" / "killed"
    ward_type: str
    creator_champion: str


@dataclass(frozen=True)
class MatchMeta:
    """試合メタ情報."""

    match_id: str
    game_duration_ms: int
    win: bool
    player_champion: str
    player_role: str


@dataclass(frozen=True)
class RawMatchData:
    """Stage A で取得した全試合データ（正規化済み）."""

    meta: MatchMeta
    participants: tuple[ParticipantStats, ...]
    kill_events: tuple[KillEvent, ...]
    objective_events: tuple[ObjectiveEvent, ...]
    building_events: tuple[BuildingEvent, ...]
    ward_events: tuple[WardEvent, ...]


@dataclass(frozen=True)
class MatchInfo:
    """Riot API から取得したマッチ情報."""

    match_id: str                          # 例: "JP1_570793513"
    classification: str                    # "full" / "unclassified"
    participants: tuple[Participant, ...]   # 10 人分
    player_champion: str | None            # プレイヤー本人のチャンピオン名

    @staticmethod
    def unavailable() -> "MatchInfo":
        """Riot API 未使用時のダミー（エラー停止前のテスト用）."""
        return MatchInfo(
            match_id="",
            classification="unavailable",
            participants=(),
            player_champion=None,
        )


COACHING_CONTEXT_SCHEMA_VERSION = "1.2"


@dataclass(frozen=True)
class CoachingContext:
    """事前分析フェーズの出力（中間生成物）.

    動画分析結果を保持し JSON で永続化する。
    """

    schema_version: str
    video_source: str
    analyzed_at: str  # ISO 8601
    match_info: MatchInfo
    raw_match_data: RawMatchData | None  # Phase 1: Riot API 生データ
    deaths: tuple[DeathScene, ...]
    good_plays: tuple[GoodPlay, ...]
    analysis: Analysis
    system_instruction: str

    @classmethod
    def from_observation_analysis(
        cls,
        video_source: str,
        match_info: MatchInfo,
        obs: Observation,
        analysis: Analysis,
        system_instruction: str,
        raw_match_data: RawMatchData | None = None,
    ) -> "CoachingContext":
        """Observation + Analysis から CoachingContext を生成する."""
        return cls(
            schema_version=COACHING_CONTEXT_SCHEMA_VERSION,
            video_source=video_source,
            analyzed_at=datetime.now(tz=timezone.utc).isoformat(),
            match_info=match_info,
            raw_match_data=raw_match_data,
            deaths=obs.deaths,
            good_plays=obs.good_plays,
            analysis=analysis,
            system_instruction=system_instruction,
        )

    def to_json(self) -> str:
        """CoachingContext を JSON 文字列に変換する."""
        d = dataclasses.asdict(self)
        d["match_info"] = {
            "match_id": self.match_info.match_id,
            "classification": self.match_info.classification,
            "participants": [dataclasses.asdict(p) for p in self.match_info.participants],
            "player_champion": self.match_info.player_champion,
        }
        if self.raw_match_data is not None:
            d["raw_match_data"] = _raw_match_data_to_dict(self.raw_match_data)
        else:
            d["raw_match_data"] = None
        d["deaths"] = [dataclasses.asdict(x) for x in self.deaths]
        d["good_plays"] = [dataclasses.asdict(x) for x in self.good_plays]
        d["analysis"] = dataclasses.asdict(self.analysis)
        return json.dumps(d, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "CoachingContext":
        """JSON 文字列から CoachingContext を復元する."""
        d = json.loads(json_str)
        if d.get("schema_version") != COACHING_CONTEXT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version mismatch: {d.get('schema_version')} != {COACHING_CONTEXT_SCHEMA_VERSION}"
            )
        raw_data = d.get("raw_match_data")
        raw_match_data = _raw_match_data_from_dict(raw_data) if raw_data else None
        mi = d["match_info"]
        match_info = MatchInfo(
            match_id=mi["match_id"],
            classification=mi["classification"],
            participants=tuple(Participant(**p) for p in mi["participants"]),
            player_champion=mi.get("player_champion"),
        )
        deaths = tuple(DeathScene(**item) for item in d["deaths"])
        good_plays = tuple(GoodPlay(**item) for item in d["good_plays"])
        a = d["analysis"]
        analysis = Analysis(
            strength=a["strength"],
            weakness=a["weakness"],
            good_scenes=tuple(a["good_scenes"]),
            bad_scenes=tuple(a["bad_scenes"]),
            good_count=a["good_count"],
            total_deaths=a["total_deaths"],
            main_weakness_cause=a["main_weakness_cause"],
        )
        return cls(
            schema_version=d["schema_version"],
            video_source=d["video_source"],
            analyzed_at=d["analyzed_at"],
            match_info=match_info,
            raw_match_data=raw_match_data,
            deaths=deaths,
            good_plays=good_plays,
            analysis=analysis,
            system_instruction=d["system_instruction"],
        )


def _raw_match_data_to_dict(raw: RawMatchData) -> dict:
    """RawMatchData を JSON シリアライズ可能な dict に変換する."""
    return {
        "meta": dataclasses.asdict(raw.meta),
        "participants": [dataclasses.asdict(p) for p in raw.participants],
        "kill_events": [dataclasses.asdict(e) for e in raw.kill_events],
        "objective_events": [dataclasses.asdict(e) for e in raw.objective_events],
        "building_events": [dataclasses.asdict(e) for e in raw.building_events],
        "ward_events": [dataclasses.asdict(e) for e in raw.ward_events],
    }


def _raw_match_data_from_dict(d: dict) -> RawMatchData:
    """dict から RawMatchData を復元する."""
    return RawMatchData(
        meta=MatchMeta(**d["meta"]),
        participants=tuple(ParticipantStats(**p) for p in d["participants"]),
        kill_events=tuple(
            KillEvent(**{**e, "assist_champions": tuple(e.get("assist_champions", ()))})
            for e in d["kill_events"]
        ),
        objective_events=tuple(ObjectiveEvent(**e) for e in d["objective_events"]),
        building_events=tuple(BuildingEvent(**e) for e in d["building_events"]),
        ward_events=tuple(WardEvent(**e) for e in d["ward_events"]),
    )


@dataclass
class ReplayDesiredState:
    """Python sidecar が望む Replay の状態.

    None は「変更しない」を意味する。
    """

    time_seconds: float | None = None
    paused: bool | None = None
    speed: float = 1.0


@dataclass(frozen=True)
class SubtitleEvent:
    """字幕イベント（Tauri フロントエンドへの通知用）."""

    text: str
    timestamp: float  # Unix timestamp
    is_user: bool  # True: ユーザー発話、False: AI 発話
    finished: bool = False  # True: このターンの字幕確定
