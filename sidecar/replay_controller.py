"""Riot Replay API 制御コントローラー.

desired_state（望む状態）と実状態を突き合わせ、ズレを補正する直列コントローラー。
barge-in や tool_call_cancellation 受信時に reconcile_now() で即時補正する。

前提:
  - LoL クライアントが localhost:2999 で Replay API を公開していること
  - game.cfg に [General] EnableReplayApi=1 が設定済みであること
  - 接続不可時は warning ログのみ（crash しない）
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import requests
import urllib3

from sidecar.models import ReplayDesiredState

logger = logging.getLogger(__name__)

# Replay API のベース URL（self-signed cert のため verify=False）
REPLAY_API_BASE = "https://127.0.0.1:2999"

# 時刻補正の閾値（秒）: 実状態とのズレがこの値を超えたら補正コマンドを再発行する
TIME_CORRECTION_THRESHOLD = 0.5

# 再生速度の安全範囲
SPEED_MIN = 0.1
SPEED_MAX = 8.0


class ReplayStateController:
    """desired_state と実状態を突き合わせ補正する直列コントローラー.

    設計: spec の desired_state パターンを実装。
      - tool_call ごとに apply_and_reconcile() を呼び出す
      - barge-in / tool_call_cancellation では reconcile_now() を呼び出す
      - asyncio.Lock で直列化し、同時リクエストによる状態競合を防ぐ
    """

    def __init__(self, http_session: requests.Session | None = None) -> None:
        """初期化.

        Args:
            http_session: 注入する HTTP セッション。None の場合は self-signed 対応セッションを生成。
                          テストではモックセッションを注入する。
        """
        self._desired = ReplayDesiredState()
        self._session = (
            http_session if http_session is not None else self._make_session()
        )
        self._lock: asyncio.Lock | None = None
        self._available = True  # Replay API の接続状態

    def _get_lock(self) -> asyncio.Lock:
        """現在のイベントループ上で利用する Lock を遅延生成する."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @staticmethod
    def _make_session() -> requests.Session:
        """self-signed 証明書を許容する requests.Session を生成する."""
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        session = requests.Session()
        session.verify = False
        return session

    @property
    def available(self) -> bool:
        """Replay API が利用可能かどうか."""
        return self._available

    def set_camera_to_player(self, champion_name: str) -> None:
        """カメラを指定チャンピオンに固定する（自分視点）.

        Args:
            champion_name: チャンピオン名（例: "Sona"）。Replay API の selectionName に渡す。
        """
        url = f"{REPLAY_API_BASE}/replay/render"
        try:
            resp = self._session.post(
                url,
                json={
                    "selectionName": champion_name,
                    "cameraAttached": True,
                },
                timeout=(0.5, 1.0),
            )
            resp.raise_for_status()
            logger.info("カメラ固定: %s", champion_name)
        except Exception:
            logger.warning("カメラ設定失敗: %s", champion_name, exc_info=True)

    def update_desired(self, **kwargs: Any) -> None:
        """desired_state を更新する.

        barge-in や新規 tool_call 時に呼び出す。
        None を渡したフィールドは「変更しない」として扱われる（モデルの定義に従う）。

        Args:
            **kwargs: ReplayDesiredState のフィールド（time_seconds, paused, speed）。
        """
        for k, v in kwargs.items():
            if hasattr(self._desired, k):
                setattr(self._desired, k, v)
            else:
                logger.warning("desired_state に未知のフィールド: %s", k)

    async def apply_and_reconcile(self, tool_name: str, args: dict[str, Any]) -> None:
        """Replay API に tool を発行し、完了後に desired_state と実状態を突き合わせる.

        ロック取得 → API 発行 → reconcile の順序を守ることで直列化を保証する。
        API 発行に失敗した場合は例外を呼び出し元へ返し、成功と区別できるようにする。

        Args:
            tool_name: ツール名（seek_replay, pause_replay, resume_replay, slow_motion）。
            args: ツール引数辞書。
        """
        async with self._get_lock():
            await asyncio.to_thread(self._call_replay_api, tool_name, args)
            await self._reconcile()

    async def reconcile_now(self) -> None:
        """barge-in 時など、tool_call 外から即時補正を呼ぶ用途.

        ロック取得後に実状態を取得し、desired_state とのズレを補正する。
        C1 仕様準拠: tool_call_cancellation を待たずに呼び出す。
        """
        async with self._get_lock():
            await self._reconcile()

    def _call_replay_api(self, tool_name: str, args: dict[str, Any]) -> None:
        """Replay API に HTTP POST を送信する（blocking, to_thread() で実行）.

        接続不可・エラー時は warning を残しつつ例外を送出し、
        呼び出し元が tool error response を返せるようにする。

        Args:
            tool_name: ツール名。
            args: ツール引数辞書。
        """
        payload = self._build_payload(tool_name, args)
        if payload is None:
            return  # 未知のツール名

        try:
            url = f"{REPLAY_API_BASE}/replay/playback"
            resp = self._session.post(url, json=payload, timeout=(0.5, 1.0))
            resp.raise_for_status()
            self._available = True
            logger.info("Replay API: %s → payload=%s", tool_name, payload)
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as e:
            self._available = False
            message = "Replay API 接続不可（LoL リプレイ起動中か確認してください）"
            logger.warning(message)
            raise RuntimeError(message) from e
        except requests.exceptions.RequestException as e:
            logger.warning("Replay API エラー: %s", e)
            raise RuntimeError(f"Replay API エラー: {e}") from e

    @staticmethod
    def _build_payload(tool_name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        """ツール名・引数から Replay API リクエストボディを生成する.

        Args:
            tool_name: ツール名。
            args: ツール引数辞書。

        Returns:
            POST ボディ辞書。未知のツール名の場合は None。
        """
        if tool_name == "seek_replay":
            return {"time": float(args.get("time_seconds", 0))}
        if tool_name == "pause_replay":
            return {"paused": True}
        if tool_name == "resume_replay":
            return {"paused": False}
        if tool_name == "slow_motion":
            speed = float(args.get("speed", 0.5))
            return {"speed": max(SPEED_MIN, min(speed, SPEED_MAX))}

        logger.warning("未知のツール名: %s", tool_name)
        return None

    def _get_actual_state(self) -> dict[str, Any]:
        """現在のリプレイ状態を取得する（blocking, to_thread() で実行）.

        接続不可時は空辞書を返す（crash しない）。

        Returns:
            playback 状態辞書（time, paused, speed, length, seeking）。
            接続不可・エラー時は {}。
        """
        try:
            url = f"{REPLAY_API_BASE}/replay/playback"
            resp = self._session.get(url, timeout=(0.5, 1.0))
            resp.raise_for_status()
            self._available = True
            return resp.json()
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ):
            self._available = False
            logger.warning("Replay API 接続不可: actual_state 取得スキップ")
            return {}
        except requests.exceptions.RequestException as e:
            logger.warning("Replay API actual_state 取得エラー: %s", e)
            return {}

    async def _reconcile(self) -> None:
        """実状態を取得し、desired_state とのズレを補正コマンドで再発行する.

        Replay API が接続不可の場合はスキップ（crash しない）。
        Lock は呼び出し元（apply_and_reconcile / reconcile_now）で取得済み。
        """
        actual = await asyncio.to_thread(self._get_actual_state)
        if not actual:
            return  # 接続不可時はスキップ

        corrections: dict[str, Any] = {}

        # paused のズレを検出
        if (
            self._desired.paused is not None
            and actual.get("paused") != self._desired.paused
        ):
            corrections["paused"] = self._desired.paused

        # 時刻のズレを検出（閾値以上のズレのみ補正）
        if self._desired.time_seconds is not None:
            actual_time = float(actual.get("time", 0.0))
            if (
                abs(actual_time - self._desired.time_seconds)
                > TIME_CORRECTION_THRESHOLD
            ):
                corrections["time"] = self._desired.time_seconds

        # TODO: MVP では speed 補正をスコープ外とする。
        # ハッカソン後に actual["speed"] との差分も補正対象へ追加する。

        if corrections:
            await asyncio.to_thread(self._apply_corrections, corrections)

    def _apply_corrections(self, corrections: dict[str, Any]) -> None:
        """補正コマンドを Replay API に送信する（blocking, to_thread() で実行）.

        Args:
            corrections: 補正するフィールドの辞書。
        """
        try:
            url = f"{REPLAY_API_BASE}/replay/playback"
            resp = self._session.post(url, json=corrections, timeout=(0.5, 1.0))
            resp.raise_for_status()
            logger.info("Replay API 補正送信: %s", corrections)
        except requests.exceptions.RequestException as e:
            logger.warning("Replay API 補正エラー: %s", e)
