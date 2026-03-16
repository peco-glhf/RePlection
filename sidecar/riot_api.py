"""Riot API クライアント — マッチデータ取得とシーン候補生成.

Stage A: Riot Grounding を担当する。
事実データ（チーム編成・キルイベント）を取得し、分析対象シーンを決定論的に生成する。
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Any

import requests

from sidecar.models import (
    BuildingEvent,
    KillEvent,
    MatchInfo,
    MatchMeta,
    ObjectiveEvent,
    Participant,
    ParticipantStats,
    RawMatchData,
    WardEvent,
)

logger = logging.getLogger(__name__)

RIOT_API_BASE = "https://asia.api.riotgames.com"
REQUEST_TIMEOUT = 30

# シーン候補のクリップ範囲（ミリ秒）
DEATH_CLIP_BEFORE_MS = 12_000
DEATH_CLIP_AFTER_MS = 8_000
GOOD_PLAY_CLIP_BEFORE_MS = 12_000
GOOD_PLAY_CLIP_AFTER_MS = 8_000

# 深掘り上限
MAX_DEATH_SCENES = 6
MAX_GOOD_PLAY_SCENES = 3


# フェーズ判定の閾値（ミリ秒）
EARLY_GAME_END_MS = 14 * 60 * 1000   # 14分
MID_GAME_END_MS = 25 * 60 * 1000     # 25分

# マップゾーン判定（Summoner's Rift 座標）
MAP_CENTER_X = 7400
MAP_CENTER_Y = 7400
RIVER_MARGIN = 1500
JUNGLE_MARGIN = 3000

# importance_score の加算値
SCORE_OBJECTIVE_NEARBY = 3
SCORE_BOUNTY_HIGH = 2        # bounty >= 300
SCORE_POST_DEATH_LOSS = 2    # デス後60秒以内にオブジェクト/タワー失う
SCORE_CONSECUTIVE_DEATH = 1
SCORE_SOLO_DEATH = 1
SCORE_OBJECTIVE_CONVERSION = 3  # キル後60秒以内にオブジェクト獲得
OBJECTIVE_WINDOW_MS = 60_000    # 前後60秒


@dataclass(frozen=True)
class CombatCluster:
    """連続したキルイベントを束ねた「戦闘」単位."""

    cluster_id: str
    fight_type: str  # "duel" / "skirmish" / "teamfight"
    start_ms: int
    end_ms: int
    kill_count: int
    participant_count: int


# クラスタリング閾値
CLUSTER_GAP_MS = 15_000  # 時間閾値（ミリ秒）
CLUSTER_SPATIAL_RADIUS = 3000  # 同一局地戦を想定した近接距離（マップ座標単位）


@dataclass(frozen=True)
class SceneCandidate:
    """Stage A で生成されるシーン候補（事実 + 軽量 derived）."""

    scene_id: str
    scene_type: str  # "death" / "good_play"
    match_time_ms: int
    actor_champion: str  # death: 殺した敵 / good_play: 自分
    victim_champion: str  # death: 自分 / good_play: 殺した敵
    assist_champions: tuple[str, ...]
    clip_start_ms: int
    clip_end_ms: int
    # 軽量 derived（Phase 2）
    phase: str             # "early" / "mid" / "late"
    map_zone: str          # "lane" / "river" / "ally_jungle" / "enemy_jungle" / "base"
    gold_diff_at_time: int # チームゴールド差（正=有利、負=不利）
    importance_score: int  # シーンの重要度スコア
    objective_context: str # 例: "dragon_before_30s" / "baron_after_45s" / ""
    bounty: int            # バウンティ額
    fight_context: str     # 例: "teamfight(5kills,8players)" / "solo"
    death_cost_score: int  # デスのコスト（高い=チーム損害大）。death のみ使用、good_play では 0
    kill_value_score: int  # キルの価値（高い=チーム貢献大）。good_play のみ使用、death では 0


def fetch_match_data(match_id: str) -> tuple[MatchInfo, list[SceneCandidate], RawMatchData]:
    """Riot API からマッチ情報を取得し、シーン候補と全試合データを返す.

    Args:
        match_id: Riot match ID（例: "JP1_570793513"）

    Returns:
        (MatchInfo, SceneCandidate のリスト, RawMatchData)

    Raises:
        RuntimeError: RIOT_MATCH_ID / RIOT_API_KEY 未設定、または API 通信障害時
    """
    api_key = _require_env("RIOT_API_KEY")
    if not match_id:
        match_id = _require_env("RIOT_MATCH_ID")

    puuid = _resolve_puuid(api_key)

    # Match-V5 から全データ取得
    match_response = _fetch_match_response(match_id, api_key)
    match_info = _build_match_info(match_id, match_response, puuid)

    # Timeline から全イベント取得
    timeline_response = _fetch_timeline_response(match_id, api_key)
    participant_map = _build_participant_map(match_response)

    # RawMatchData を構築
    raw_match_data = _build_raw_match_data(
        match_id, match_response, timeline_response, participant_map, match_info, puuid,
    )

    # SceneCandidate を生成（軽量 derived 付き）
    candidates = _generate_candidates(
        raw_match_data.kill_events, participant_map, match_info, raw_match_data,
    )

    logger.info(
        "Stage A 完了: deaths=%d, good_plays=%d, kills=%d, objectives=%d",
        sum(1 for c in candidates if c.scene_type == "death"),
        sum(1 for c in candidates if c.scene_type == "good_play"),
        len(raw_match_data.kill_events),
        len(raw_match_data.objective_events),
    )
    return match_info, candidates, raw_match_data


def build_match_context_prompt(match_info: MatchInfo) -> str:
    """Gemini プロンプト注入用テキストを生成する."""
    allies = [p for p in match_info.participants if p.team == "ally"]
    enemies = [p for p in match_info.participants if p.team == "enemy"]

    lines = ["この試合のチーム編成:"]
    if allies:
        lines.append("味方: " + ", ".join(f"{p.champion}({p.role})" for p in allies))
    if enemies:
        lines.append("敵:   " + ", ".join(f"{p.champion}({p.role})" for p in enemies))
    if not allies and not enemies:
        lines.append("参加者: " + ", ".join(f"{p.champion}({p.role})" for p in match_info.participants))

    if match_info.player_champion:
        lines.append(f"分析対象プレイヤー: {match_info.player_champion}")

    lines.append("チャンピオン名は上記リストの正確な名前のみを使用してください。推測しないでください。")
    return "\n".join(lines)


def build_scene_analysis_prompt(
    candidate: SceneCandidate,
    match_info: MatchInfo,
    raw: RawMatchData | None = None,
) -> str:
    """1 シーンの分析プロンプトを生成する（事実 + ゲーム状態 + 判断評価指示）."""
    time_str = ms_to_spoken(candidate.match_time_ms)
    phase_label = {"early": "序盤", "mid": "中盤", "late": "終盤"}.get(candidate.phase, "不明")
    zone_label = {
        "lane": "レーン", "river": "川", "ally_jungle": "自陣ジャングル",
        "enemy_jungle": "敵ジャングル", "base": "ベース", "unknown": "不明",
    }.get(candidate.map_zone, "不明")

    lines = _build_fact_block(candidate, time_str, phase_label, zone_label, match_info)
    lines.extend(_build_context_block(candidate, raw))
    lines.extend(_build_analysis_instruction(candidate))
    lines.append("事実は上記の通り固定です。チャンピオン名を推測・変更しないでください。")
    return "\n".join(lines)


def _build_fact_block(
    candidate: SceneCandidate, time_str: str, phase_label: str,
    zone_label: str, match_info: MatchInfo,
) -> list[str]:
    """事実ブロックを構築する."""
    lines = ["【事実（変更不可）】"]
    if candidate.scene_type == "death":
        actor_role = _find_role(match_info, candidate.actor_champion)
        lines.append(f"- 時刻: {time_str}（{phase_label}）")
        lines.append(f"- {candidate.victim_champion} が {candidate.actor_champion}({actor_role}) にキルされた")
        lines.append(f"- アシスト: {', '.join(candidate.assist_champions) or 'なし'}")
        lines.append(f"- 場所: {zone_label}")
        if candidate.bounty > 0:
            lines.append(f"- バウンティ: {candidate.bounty} gold")
    else:
        actor_role = _find_role(match_info, candidate.actor_champion)
        lines.append(f"- 時刻: {time_str}（{phase_label}）")
        lines.append(f"- {candidate.actor_champion}({actor_role}) が {candidate.victim_champion} をキルした")
        lines.append(f"- アシスト: {', '.join(candidate.assist_champions) or 'なし'}")
        lines.append(f"- 場所: {zone_label}")
    return lines


def _build_context_block(candidate: SceneCandidate, raw: RawMatchData | None) -> list[str]:
    """ゲーム状態・直近イベントの文脈ブロックを構築する."""
    lines = ["\n【試合文脈】"]
    if candidate.fight_context != "solo":
        lines.append(f"- この{'死亡' if candidate.scene_type == 'death' else 'キル'}は{candidate.fight_context}の中で起きた")
    if candidate.objective_context:
        lines.append(f"- オブジェクト: {candidate.objective_context}")
    if candidate.gold_diff_at_time != 0:
        diff_label = "有利" if candidate.gold_diff_at_time > 0 else "不利"
        lines.append(f"- ゴールド差: {abs(candidate.gold_diff_at_time)} gold {diff_label}")
    if candidate.scene_type == "death" and candidate.death_cost_score >= 3:
        lines.append("- このデスはチームにとってコストが高い（オブジェクト損失あり）")
    if candidate.scene_type == "good_play" and candidate.kill_value_score >= 3:
        lines.append("- このキルはオブジェクト獲得に繋がった")

    # 直近60秒のイベント（raw がある場合）
    if raw:
        recent = _recent_events_near(candidate.match_time_ms, raw)
        if recent:
            lines.append("\n【直近60秒のイベント】")
            lines.extend(recent)
    return lines


def _recent_events_near(timestamp_ms: int, raw: RawMatchData) -> list[str]:
    """直近60秒のキル/オブジェクトイベントを返す."""
    items: list[tuple[int, str]] = []
    for ev in raw.kill_events:
        delta = timestamp_ms - ev.timestamp_ms
        if 0 < delta <= 60_000:
            items.append((delta, f"- {delta // 1000}秒前: {ev.killer_champion} が {ev.victim_champion} をキル"))
    for ev in raw.objective_events:
        delta = timestamp_ms - ev.timestamp_ms
        if 0 < delta <= 60_000:
            team = "味方" if ev.killer_team == "ally" else "敵"
            items.append((delta, f"- {delta // 1000}秒前: {team}が {ev.monster_type} 獲得"))
    items.sort(key=lambda x: x[0], reverse=True)
    return [line for _, line in items[:5]]


def _build_analysis_instruction(candidate: SceneCandidate) -> list[str]:
    """分析指示ブロックを構築する."""
    clip_range = f"{_ms_to_mmss(candidate.clip_start_ms)}〜{_ms_to_mmss(candidate.clip_end_ms)}"
    lines: list[str] = []
    if candidate.scene_type == "death":
        lines.append(f"\n動画の {clip_range} を見て、以下を分析してください:")
        lines.append("1. デスの15秒前、プレイヤーはどんな判断をしたか")
        lines.append("2. その判断は上記の状況で合理的だったか")
        lines.append("3. どの時点でデスを避けられた最後のチャンスがあったか")
        lines.append("4. 同じ状況で取るべきだった行動")
        lines.append("")
        lines.append("注意: 相関ではなく因果を特定せよ。")
        lines.append("「デスの直前にしていた行動」が必ずしも原因ではない。")
        lines.append("「その行動を変えていれば、このデスは回避できたか？」を判定基準にせよ。")
    else:
        lines.append(f"\n動画の {clip_range} を見て、以下を分析してください:")
        lines.append("1. このプレイの成功要因（タイミング・ポジショニング・判断）")
        lines.append("2. このプレイのリスクは何だったか（失敗していたらどうなったか）")
        lines.append("3. 同じ状況で再現するために必要な条件")
    return lines


def cache_fingerprint(match_id: str, puuid: str | None) -> str:
    """キャッシュキー用のフィンガープリント."""
    raw = f"{match_id}:{puuid or 'no_player'}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# 内部関数
# ---------------------------------------------------------------------------


def _get_video_offset_ms() -> int:
    """動画とゲーム時刻のオフセット（ミリ秒）を取得する.

    正の値: 動画がゲーム開始より遅れて始まる（例: 5000 = ゲーム5秒後から録画開始）
    変換式: video_ms = match_time_ms - VIDEO_OFFSET_MS

    Returns:
        オフセット値（ミリ秒）。無効な値の場合は 0
    """
    raw = os.getenv("VIDEO_OFFSET_MS", "0")
    try:
        return int(raw)
    except ValueError:
        logger.warning("VIDEO_OFFSET_MS の値が不正です（数値でない）: %r → 0 を使用", raw)
        return 0


def _require_env(name: str) -> str:
    """必須環境変数を取得する."""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} が未設定です")
    return value


def _resolve_puuid(api_key: str) -> str | None:
    """PUUID を解決する（.env から直接 or Riot ID から取得）."""
    puuid = os.getenv("RIOT_PUUID", "").strip()
    if puuid:
        return puuid

    game_name = os.getenv("RIOT_GAME_NAME", "").strip()
    tag_line = os.getenv("RIOT_TAG_LINE", "").strip()
    if game_name and tag_line:
        logger.info("PUUID 取得中: %s#%s", game_name, tag_line)
        url = f"{RIOT_API_BASE}/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
        resp = requests.get(url, headers=_headers(api_key), timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        resolved = resp.json()["puuid"]
        logger.info("PUUID 取得完了: %s...", resolved[:8])
        return resolved

    logger.warning("プレイヤー識別情報なし（RIOT_PUUID / RIOT_GAME_NAME+RIOT_TAG_LINE）")
    return None


def _headers(api_key: str) -> dict[str, str]:
    return {"X-Riot-Token": api_key}


def _fetch_match_response(match_id: str, api_key: str) -> dict[str, Any]:
    """Match-V5 API のレスポンスを取得する."""
    logger.info("マッチ情報取得中: %s", match_id)
    url = f"{RIOT_API_BASE}/lol/match/v5/matches/{match_id}"
    resp = requests.get(url, headers=_headers(api_key), timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _fetch_timeline_response(match_id: str, api_key: str) -> dict[str, Any]:
    """Timeline API のレスポンスを取得する."""
    logger.info("Timeline 取得中: %s", match_id)
    url = f"{RIOT_API_BASE}/lol/match/v5/matches/{match_id}/timeline"
    resp = requests.get(url, headers=_headers(api_key), timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _build_participant_map(match_response: dict[str, Any]) -> dict[int, str]:
    """participantId → championName のマッピングを構築する."""
    participants = match_response.get("info", {}).get("participants", [])
    return {p["participantId"]: p["championName"] for p in participants}


def _build_match_info(
    match_id: str, match_response: dict[str, Any], puuid: str | None,
) -> MatchInfo:
    """Match レスポンスから MatchInfo を構築する."""
    participants_raw = match_response.get("info", {}).get("participants", [])

    player_team_id = None
    player_champion = None
    if puuid:
        for p in participants_raw:
            if p.get("puuid") == puuid:
                player_team_id = p.get("teamId")
                player_champion = p.get("championName")
                break

    participants: list[Participant] = []
    for p in participants_raw:
        if puuid and player_team_id is not None:
            team = "ally" if p.get("teamId") == player_team_id else "enemy"
        else:
            team = "unknown"
        participants.append(Participant(
            champion=p.get("championName", "Unknown"),
            role=p.get("teamPosition", "UNKNOWN"),
            team=team,
        ))

    classification = "full" if puuid and player_team_id is not None else "unclassified"
    logger.info("チーム編成取得完了: classification=%s, player=%s", classification, player_champion)
    return MatchInfo(
        match_id=match_id,
        classification=classification,
        participants=tuple(participants),
        player_champion=player_champion,
    )


def _build_raw_match_data(
    match_id: str,
    match_response: dict[str, Any],
    timeline_response: dict[str, Any],
    participant_map: dict[int, str],
    match_info: MatchInfo,
    puuid: str | None,
) -> RawMatchData:
    """Match + Timeline レスポンスから RawMatchData を構築する."""
    info = match_response.get("info", {})
    participants_raw = info.get("participants", [])

    # MatchMeta
    player_team_id = None
    player_role = ""
    win = False
    if puuid:
        for p in participants_raw:
            if p.get("puuid") == puuid:
                player_team_id = p.get("teamId")
                player_role = p.get("teamPosition", "")
                win = bool(p.get("win", False))
                break

    meta = MatchMeta(
        match_id=match_id,
        game_duration_ms=info.get("gameDuration", 0) * 1000,
        win=win,
        player_champion=match_info.player_champion or "",
        player_role=player_role,
    )

    # ParticipantStats（10人分）
    stats: list[ParticipantStats] = []
    for p in participants_raw:
        if puuid and player_team_id is not None:
            team = "ally" if p.get("teamId") == player_team_id else "enemy"
        else:
            team = "unknown"
        stats.append(ParticipantStats(
            champion=p.get("championName", "Unknown"),
            role=p.get("teamPosition", "UNKNOWN"),
            team=team,
            kills=p.get("kills", 0),
            deaths=p.get("deaths", 0),
            assists=p.get("assists", 0),
            total_damage_dealt=p.get("totalDamageDealtToChampions", 0),
            total_damage_taken=p.get("totalDamageTaken", 0),
            total_cs=p.get("totalMinionsKilled", 0) + p.get("neutralMinionsKilled", 0),
            gold_earned=p.get("goldEarned", 0),
            vision_score=p.get("visionScore", 0),
            wards_placed=p.get("wardsPlaced", 0),
            wards_killed=p.get("wardsKilled", 0),
        ))

    # Timeline イベント抽出
    kill_events: list[KillEvent] = []
    objective_events: list[ObjectiveEvent] = []
    building_events: list[BuildingEvent] = []
    ward_events: list[WardEvent] = []

    for frame in timeline_response.get("info", {}).get("frames", []):
        for event in frame.get("events", []):
            event_type = event.get("type", "")

            if event_type == "CHAMPION_KILL":
                pos = event.get("position", {})
                assists = event.get("assistingParticipantIds", [])
                kill_events.append(KillEvent(
                    timestamp_ms=event.get("timestamp", 0),
                    killer_champion=participant_map.get(event.get("killerId", 0), ""),
                    victim_champion=participant_map.get(event.get("victimId", 0), ""),
                    assist_champions=tuple(
                        participant_map.get(aid, "") for aid in assists if participant_map.get(aid)
                    ),
                    position_x=pos.get("x", 0),
                    position_y=pos.get("y", 0),
                    bounty=event.get("bounty", 0),
                ))

            elif event_type == "ELITE_MONSTER_KILL":
                killer_team_id = event.get("killerTeamId", 0)
                if player_team_id is not None:
                    killer_team = "ally" if killer_team_id == player_team_id else "enemy"
                else:
                    killer_team = "unknown"
                objective_events.append(ObjectiveEvent(
                    timestamp_ms=event.get("timestamp", 0),
                    monster_type=event.get("monsterType", "UNKNOWN"),
                    killer_team=killer_team,
                ))

            elif event_type == "BUILDING_KILL":
                lost_team_id = event.get("teamId", 0)
                if player_team_id is not None:
                    lost_team = "ally" if lost_team_id == player_team_id else "enemy"
                else:
                    lost_team = "unknown"
                building_events.append(BuildingEvent(
                    timestamp_ms=event.get("timestamp", 0),
                    building_type=event.get("buildingType", "UNKNOWN"),
                    lost_team=lost_team,
                ))

            elif event_type in ("WARD_PLACED", "WARD_KILL"):
                creator_id = event.get("creatorId", 0)
                ward_events.append(WardEvent(
                    timestamp_ms=event.get("timestamp", 0),
                    event_type="placed" if event_type == "WARD_PLACED" else "killed",
                    ward_type=event.get("wardType", "UNKNOWN"),
                    creator_champion=participant_map.get(creator_id, ""),
                ))

    logger.info(
        "RawMatchData 構築完了: kills=%d, objectives=%d, buildings=%d, wards=%d",
        len(kill_events), len(objective_events), len(building_events), len(ward_events),
    )
    return RawMatchData(
        meta=meta,
        participants=tuple(stats),
        kill_events=tuple(kill_events),
        objective_events=tuple(objective_events),
        building_events=tuple(building_events),
        ward_events=tuple(ward_events),
    )


def _build_combat_clusters(
    kill_events: tuple[KillEvent, ...],
) -> list[CombatCluster]:
    """キルイベントを時間的近接性で連結し、戦闘クラスタを構築する.

    Args:
        kill_events: 時系列順のキルイベント

    Returns:
        CombatCluster のリスト（時系列順）
    """
    if not kill_events:
        return []

    sorted_events = sorted(kill_events, key=lambda e: e.timestamp_ms)

    def _event_participants(ev: KillEvent) -> set[str]:
        s: set[str] = set()
        if ev.killer_champion:
            s.add(ev.killer_champion)
        if ev.victim_champion:
            s.add(ev.victim_champion)
        for a in ev.assist_champions:
            if a:
                s.add(a)
        return s

    def _group_participants(group: list[KillEvent]) -> set[str]:
        s: set[str] = set()
        for ev in group:
            s |= _event_participants(ev)
        return s

    def _spatial_proximity(ev: KillEvent, group: list[KillEvent]) -> bool:
        """イベントとグループ内の直近キルが近い位置か（距離 3000 以内）."""
        last = group[-1]
        dx = ev.position_x - last.position_x
        dy = ev.position_y - last.position_y
        return (dx * dx + dy * dy) < CLUSTER_SPATIAL_RADIUS * CLUSTER_SPATIAL_RADIUS

    # connected components: Δt <= CLUSTER_GAP_MS かつ (参加者重なり OR 位置近接) なら同一クラスタ
    groups: list[list[KillEvent]] = [[sorted_events[0]]]
    for event in sorted_events[1:]:
        last_group = groups[-1]
        time_close = event.timestamp_ms - last_group[-1].timestamp_ms <= CLUSTER_GAP_MS
        shared = bool(_event_participants(event) & _group_participants(last_group))
        nearby = _spatial_proximity(event, last_group)
        if time_close and (shared or nearby):
            last_group.append(event)
        else:
            groups.append([event])

    clusters: list[CombatCluster] = []
    for idx, group in enumerate(groups):
        kill_count = len(group)
        # 参加者: killer + victim + assist 全員のユニーク数
        participants: set[str] = set()
        for ev in group:
            if ev.killer_champion:
                participants.add(ev.killer_champion)
            if ev.victim_champion:
                participants.add(ev.victim_champion)
            for a in ev.assist_champions:
                if a:
                    participants.add(a)
        participant_count = len(participants)

        # fight_type 分類
        if kill_count >= 3 and participant_count >= 6:
            fight_type = "teamfight"
        elif kill_count >= 2 or 3 <= participant_count <= 6:
            fight_type = "skirmish"
        else:
            # kill_count=1, participant_count<=3
            fight_type = "duel"

        clusters.append(CombatCluster(
            cluster_id=f"cluster_{idx:03d}",
            fight_type=fight_type,
            start_ms=group[0].timestamp_ms,
            end_ms=group[-1].timestamp_ms,
            kill_count=kill_count,
            participant_count=participant_count,
        ))

    return clusters


def _fight_context_for_event(
    timestamp_ms: int,
    clusters: list[CombatCluster],
) -> str:
    """キルイベントの timestamp から fight_context 文字列を生成する.

    Args:
        timestamp_ms: キルイベントのタイムスタンプ
        clusters: 構築済みの CombatCluster リスト

    Returns:
        例: "teamfight(5kills,8players)" / "duel" / "solo"
    """
    for cluster in clusters:
        if cluster.start_ms <= timestamp_ms <= cluster.end_ms:
            if cluster.fight_type == "duel" and cluster.kill_count == 1:
                return "duel"
            return (
                f"{cluster.fight_type}"
                f"({cluster.kill_count}kills,{cluster.participant_count}players)"
            )
    return "solo"


def _generate_candidates(
    kill_events: tuple[KillEvent, ...],
    participant_map: dict[int, str],
    match_info: MatchInfo,
    raw: RawMatchData | None = None,
) -> list[SceneCandidate]:
    """キルイベントから death + good_play の SceneCandidate を生成する."""
    deaths: list[SceneCandidate] = []
    good_plays: list[SceneCandidate] = []
    objective_events = raw.objective_events if raw else ()
    building_events = raw.building_events if raw else ()
    offset = _get_video_offset_ms()

    # 戦闘クラスタを構築
    clusters = _build_combat_clusters(kill_events)

    # 自分の直前デス時刻（連続デス検出用）
    prev_death_ms: int | None = None

    for event in kill_events:
        killer_champ = event.killer_champion
        victim_champ = event.victim_champion
        ts = event.timestamp_ms

        # 自分のデス
        if match_info.player_champion and victim_champ and victim_champ == match_info.player_champion:
            score = 0
            obj_ctx = _objective_context(ts, objective_events, match_info)
            if obj_ctx:
                score += SCORE_OBJECTIVE_NEARBY
            if event.bounty >= 300:
                score += SCORE_BOUNTY_HIGH
            if _has_post_death_loss(ts, building_events, objective_events, match_info):
                score += SCORE_POST_DEATH_LOSS
            if prev_death_ms is not None and ts - prev_death_ms < 120_000:
                score += SCORE_CONSECUTIVE_DEATH
            if not event.assist_champions:
                score += SCORE_SOLO_DEATH

            # death_cost_score: デスのチームへのコスト
            death_cost = 0
            if _has_post_death_loss(ts, building_events, objective_events, match_info):
                death_cost += 3
            if event.bounty >= 300:
                death_cost += 2

            deaths.append(SceneCandidate(
                scene_id=f"death_{len(deaths):03d}",
                scene_type="death",
                match_time_ms=ts,
                actor_champion=killer_champ,
                victim_champion=victim_champ,
                assist_champions=event.assist_champions,
                clip_start_ms=max(0, ts - offset - DEATH_CLIP_BEFORE_MS),
                clip_end_ms=max(1000, ts - offset + DEATH_CLIP_AFTER_MS),
                phase=_classify_phase(ts),
                map_zone=_classify_map_zone(event.position_x, event.position_y),
                gold_diff_at_time=0,  # Phase 2 では frames 未使用、後続で改善
                importance_score=score,
                objective_context=obj_ctx,
                bounty=event.bounty,
                fight_context=_fight_context_for_event(ts, clusters),
                death_cost_score=death_cost,
                kill_value_score=0,
            ))
            prev_death_ms = ts

        # 自分のキル（good_play 候補）
        if match_info.player_champion and killer_champ and killer_champ == match_info.player_champion:
            score = 0
            obj_ctx = _objective_context(ts, objective_events, match_info)
            if _has_objective_conversion(ts, objective_events, match_info):
                score += SCORE_OBJECTIVE_CONVERSION
            if obj_ctx:
                score += SCORE_OBJECTIVE_NEARBY

            # kill_value_score: キルのチームへの貢献
            kill_value = 0
            if _has_objective_conversion(ts, objective_events, match_info):
                kill_value += 3
            if event.bounty >= 300:
                kill_value += 2
            if len(event.assist_champions) >= 4:
                kill_value -= 1

            good_plays.append(SceneCandidate(
                scene_id=f"good_{len(good_plays):03d}",
                scene_type="good_play",
                match_time_ms=ts,
                actor_champion=killer_champ,
                victim_champion=victim_champ,
                assist_champions=event.assist_champions,
                clip_start_ms=max(0, ts - offset - GOOD_PLAY_CLIP_BEFORE_MS),
                clip_end_ms=max(1000, ts - offset + GOOD_PLAY_CLIP_AFTER_MS),
                phase=_classify_phase(ts),
                map_zone=_classify_map_zone(event.position_x, event.position_y),
                gold_diff_at_time=0,
                importance_score=score,
                objective_context=obj_ctx,
                bounty=event.bounty,
                fight_context=_fight_context_for_event(ts, clusters),
                death_cost_score=0,
                kill_value_score=kill_value,
            ))

    # importance_score でソートして上限で絞る
    deaths.sort(key=lambda c: c.importance_score, reverse=True)
    # 明確な低価値キル（cleanup: アシスト4+かつスコア負）のみ除外。中立キルは残す
    good_plays = [g for g in good_plays if g.kill_value_score >= 0]
    good_plays.sort(key=lambda c: c.importance_score, reverse=True)
    return deaths[:MAX_DEATH_SCENES] + good_plays[:MAX_GOOD_PLAY_SCENES]


def _classify_phase(timestamp_ms: int) -> str:
    """ゲーム時刻からフェーズを判定する."""
    if timestamp_ms < EARLY_GAME_END_MS:
        return "early"
    if timestamp_ms < MID_GAME_END_MS:
        return "mid"
    return "late"


def _classify_map_zone(x: int, y: int) -> str:
    """マップ座標からゾーンを判定する."""
    if x == 0 and y == 0:
        return "unknown"
    dx = abs(x - MAP_CENTER_X)
    dy = abs(y - MAP_CENTER_Y)
    if dx + dy < RIVER_MARGIN:
        return "river"
    if x < MAP_CENTER_X - JUNGLE_MARGIN or y < MAP_CENTER_Y - JUNGLE_MARGIN:
        return "ally_jungle" if x < MAP_CENTER_X else "enemy_jungle"
    if x > MAP_CENTER_X + JUNGLE_MARGIN or y > MAP_CENTER_Y + JUNGLE_MARGIN:
        return "enemy_jungle" if x > MAP_CENTER_X else "ally_jungle"
    return "lane"


def _objective_context(
    timestamp_ms: int,
    objectives: tuple[ObjectiveEvent, ...],
    match_info: MatchInfo,
) -> str:
    """前後60秒以内のオブジェクトイベントを文字列化する."""
    for obj in objectives:
        delta = obj.timestamp_ms - timestamp_ms
        if abs(delta) <= OBJECTIVE_WINDOW_MS:
            direction = "before" if delta < 0 else "after"
            seconds = abs(delta) // 1000
            return f"{obj.monster_type}_{direction}_{seconds}s"
    return ""


def _has_post_death_loss(
    death_ms: int,
    buildings: tuple[BuildingEvent, ...],
    objectives: tuple[ObjectiveEvent, ...],
    match_info: MatchInfo,
) -> bool:
    """デス後60秒以内に味方がオブジェクト/タワーを失ったか."""
    for b in buildings:
        if b.lost_team == "ally" and 0 < b.timestamp_ms - death_ms <= OBJECTIVE_WINDOW_MS:
            return True
    for o in objectives:
        if o.killer_team == "enemy" and 0 < o.timestamp_ms - death_ms <= OBJECTIVE_WINDOW_MS:
            return True
    return False


def _has_objective_conversion(
    kill_ms: int,
    objectives: tuple[ObjectiveEvent, ...],
    match_info: MatchInfo,
) -> bool:
    """キル後60秒以内に味方がオブジェクトを獲得したか."""
    for o in objectives:
        if o.killer_team == "ally" and 0 < o.timestamp_ms - kill_ms <= OBJECTIVE_WINDOW_MS:
            return True
    return False


def _find_role(match_info: MatchInfo, champion_name: str) -> str:
    """チャンピオン名からロールを返す."""
    for p in match_info.participants:
        if p.champion == champion_name:
            return p.role
    return "UNKNOWN"


def _ms_to_mmss(ms: int) -> str:
    """ミリ秒を MM:SS 形式に変換する（機械用: replay_window 等）."""
    total_seconds = ms // 1000
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


def ms_to_spoken(ms: int) -> str:
    """ミリ秒を音声読み上げ用フォーマットに変換する（例: 11分30秒）."""
    total_seconds = ms // 1000
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    if seconds == 0:
        return f"{minutes}分"
    return f"{minutes}分{seconds}秒"
