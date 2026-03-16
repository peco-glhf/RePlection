/**
 * RePlection — Companion Window main script.
 *
 * Polls sidecar (FastAPI / port 8765) to:
 * - Update state orb (IDLE / ANALYZING / READY / LISTENING / PROCESSING / SPEAKING / INTERRUPTED)
 * - Control Analyze / Start / Stop session
 * - Push-to-talk toggle
 */

import { getCurrentWindow } from "@tauri-apps/api/window";
import { open } from "@tauri-apps/plugin-dialog";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SIDECAR_BASE = "http://localhost:8765";
const POLL_INTERVAL_STATE = 300; // ms
const FETCH_TIMEOUT_MS = 2000; // ms
const CONNECTION_FAILURE_THRESHOLD = 3;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type CoachState = "idle" | "analyzing" | "ready" | "dialogue" | "error";
type DialogueState =
  | "idle"
  | "listening"
  | "user_speaking"
  | "processing"
  | "speaking"
  | "interrupted";

interface StateResponse {
  coach_state: CoachState;
  dialogue_state: DialogueState;
  context_path?: string;
}

type OrbState =
  | "idle"
  | "analyzing"
  | "ready"
  | "listening"
  | "user_speaking"
  | "speaking"
  | "processing"
  | "interrupted"
  | "error";

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------

const orbEl = document.getElementById("orb")!;
const orbRingEl = document.getElementById("orbRing")!;
const stateLabelEl = document.getElementById("stateLabel")!;
const startBtnEl = document.getElementById("startBtn") as HTMLButtonElement;
const stopBtnEl = document.getElementById("stopBtn") as HTMLButtonElement;
const pttBtnEl = document.getElementById("pttBtn") as HTMLButtonElement;
const closeBtnEl = document.getElementById("closeBtn")!;
const minBtnEl = document.getElementById("minBtn")!;
const clickBtnEl = document.getElementById("clickBtn");
const videoBtnEl = document.getElementById("videoBtn") as HTMLButtonElement;
const videoInfoEl = document.getElementById("videoInfo")!;
const videoNameEl = document.getElementById("videoName")!;
const videoClearBtnEl = document.getElementById("videoClearBtn")!;
const statusDotEl = document.getElementById("statusDot")!;
const statusTextEl = document.getElementById("statusText")!;

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

let isConnected = false;
let isClickThrough = false;
let isPushToTalkPressed = false;
let currentOrbState: OrbState = "idle";
let currentCoachState: CoachState = "idle";
let isPollingState = false;
let pttRequestToken = 0;
let connectionFailureCount = 0;
let selectedVideoPath: string | null = null;
let currentContextPath: string | null = null;

// ---------------------------------------------------------------------------
// State labels
// ---------------------------------------------------------------------------

const STATE_LABELS: Record<OrbState, string> = {
  idle: "Idle",
  analyzing: "Analyzing...",
  ready: "Ready",
  listening: "Listening",
  user_speaking: "You're talking",
  speaking: "Speaking",
  processing: "Thinking...",
  interrupted: "Interrupted",
  error: "Error",
};

// ---------------------------------------------------------------------------
// Sidecar API
// ---------------------------------------------------------------------------

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T | null> {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

  try {
    const res = await fetch(`${SIDECAR_BASE}${path}`, {
      ...init,
      signal: controller.signal,
    });
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  } finally {
    window.clearTimeout(timeoutId);
  }
}

async function fetchState(): Promise<StateResponse | null> {
  return fetchJson<StateResponse>("/state");
}

async function postAnalyze(videoPath: string): Promise<void> {
  await fetchJson("/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ video_path: videoPath }),
  });
}

async function postStart(contextPath: string): Promise<void> {
  await fetchJson("/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ context_path: contextPath }),
  });
}

async function postStop(): Promise<void> {
  await fetchJson("/stop", { method: "POST" });
}

async function postPushToTalkStart(): Promise<void> {
  await fetchJson("/ptt/start", { method: "POST" });
}

async function postPushToTalkEnd(): Promise<void> {
  await fetchJson("/ptt/end", { method: "POST" });
}

// ---------------------------------------------------------------------------
// CoachState + DialogueState → OrbState
// ---------------------------------------------------------------------------

function resolveOrbState(
  coach: CoachState,
  dialogue: DialogueState
): OrbState {
  if (coach === "analyzing") return "analyzing";
  if (coach === "ready") return "ready";
  if (coach === "error") return "error";
  if (coach !== "dialogue") return "idle";

  switch (dialogue) {
    case "listening":
      return "listening";
    case "user_speaking":
      return "user_speaking";
    case "processing":
      return "processing";
    case "speaking":
      return "speaking";
    case "interrupted":
      return "interrupted";
    default:
      return "idle";
  }
}

// ---------------------------------------------------------------------------
// UI updates
// ---------------------------------------------------------------------------

/** Update state orb (skip if same state). */
function updateOrb(orbState: OrbState): void {
  if (orbState === currentOrbState) return;

  currentOrbState = orbState;

  const ALL_STATES: OrbState[] = [
    "idle",
    "analyzing",
    "ready",
    "listening",
    "user_speaking",
    "speaking",
    "processing",
    "interrupted",
    "error",
  ];
  for (const s of ALL_STATES) {
    orbEl.classList.remove(`orb--${s}`);
  }
  orbEl.classList.add(`orb--${orbState}`);

  // Spinning ring (processing / analyzing only)
  orbRingEl.className = "orb-ring";
  if (orbState === "processing" || orbState === "analyzing") {
    orbRingEl.classList.add("orb-ring--processing");
  }

  stateLabelEl.textContent = STATE_LABELS[orbState];
}

/** Update connection status indicator. */
function updateConnectionStatus(connected: boolean): void {
  if (isConnected === connected) return;
  isConnected = connected;

  if (connected) {
    statusDotEl.className = "status-dot connected";
    statusTextEl.textContent = "Connected";
  } else {
    statusDotEl.className = "status-dot error";
    statusTextEl.textContent = "Disconnected";
    updateOrb("idle");
  }
}

/** Update Start / Stop / video button enabled state and labels. */
function updateButtons(coachState: CoachState): void {
  const videoSelected = selectedVideoPath !== null;

  // START button: idle (with video) → ANALYZE, ready → START
  if (coachState === "idle") {
    startBtnEl.disabled = !videoSelected;
    startBtnEl.textContent = "▶ ANALYZE";
  } else if (coachState === "analyzing") {
    startBtnEl.disabled = true;
    startBtnEl.textContent = "▶ ANALYZE";
  } else if (coachState === "ready") {
    startBtnEl.disabled = false;
    startBtnEl.textContent = "▶ START";
  } else {
    startBtnEl.disabled = true;
    startBtnEl.textContent = "▶ START";
  }

  // STOP: active when not idle/ready
  stopBtnEl.disabled = coachState === "idle" || coachState === "ready";

  // PTT: only during dialogue
  pttBtnEl.disabled = coachState !== "dialogue";
  pttBtnEl.classList.toggle("active", isPushToTalkPressed);

  // Video select: only when idle
  videoBtnEl.disabled = coachState !== "idle";
}

// ---------------------------------------------------------------------------
// Polling
// ---------------------------------------------------------------------------

async function pollState(): Promise<void> {
  if (isPollingState) return;
  isPollingState = true;
  const state = await fetchState();
  try {
    if (!state) {
      connectionFailureCount += 1;
      if (connectionFailureCount >= CONNECTION_FAILURE_THRESHOLD) {
        updateConnectionStatus(false);
      }
      if (isPushToTalkPressed) {
        isPushToTalkPressed = false;
        updateButtons(currentCoachState);
      }
      return;
    }
    connectionFailureCount = 0;
    currentCoachState = state.coach_state;

    // context_path を保存（READY 時のみ存在）
    if (state.context_path) {
      currentContextPath = state.context_path;
    }
    if (state.coach_state === "idle") {
      currentContextPath = null;
    }

    updateConnectionStatus(true);
    updateOrb(resolveOrbState(state.coach_state, state.dialogue_state));
    updateButtons(state.coach_state);
    if (
      isPushToTalkPressed &&
      !["user_speaking", "interrupted", "processing"].includes(
        state.dialogue_state
      )
    ) {
      isPushToTalkPressed = false;
      updateButtons(currentCoachState);
    }
  } finally {
    isPollingState = false;
  }
}

async function startPushToTalk(): Promise<void> {
  if (isPushToTalkPressed || currentCoachState !== "dialogue") return;
  const token = ++pttRequestToken;
  isPushToTalkPressed = true;
  updateButtons(currentCoachState);
  try {
    await postPushToTalkStart();
  } catch {
    if (pttRequestToken === token) {
      isPushToTalkPressed = false;
      updateButtons(currentCoachState);
    }
  }
}

async function endPushToTalk(): Promise<void> {
  if (!isPushToTalkPressed) return;
  const token = ++pttRequestToken;
  isPushToTalkPressed = false;
  updateButtons(currentCoachState);
  try {
    await postPushToTalkEnd();
  } catch {
    if (pttRequestToken === token) {
      updateButtons(currentCoachState);
    }
  }
}

// ---------------------------------------------------------------------------
// Event handlers
// ---------------------------------------------------------------------------

startBtnEl.addEventListener("click", async () => {
  if (currentCoachState === "ready" && currentContextPath) {
    // Phase 2: start live conversation
    startBtnEl.disabled = true;
    try {
      await postStart(currentContextPath);
    } catch {
      startBtnEl.disabled = false;
    }
  } else if (currentCoachState === "idle" && selectedVideoPath) {
    // Phase 1: start analysis
    startBtnEl.disabled = true;
    try {
      await postAnalyze(selectedVideoPath);
    } catch {
      startBtnEl.disabled = false;
    }
  }
});

stopBtnEl.addEventListener("click", async () => {
  stopBtnEl.disabled = true;
  isPushToTalkPressed = false;
  try {
    await postStop();
  } catch {
    stopBtnEl.disabled = false;
  }
});

pttBtnEl.addEventListener("pointerdown", (event) => {
  if (event.button !== 0) return;
  event.preventDefault();
  pttBtnEl.setPointerCapture(event.pointerId);
  void startPushToTalk();
});

pttBtnEl.addEventListener("pointerup", (event) => {
  if (pttBtnEl.hasPointerCapture(event.pointerId)) {
    pttBtnEl.releasePointerCapture(event.pointerId);
  }
  void endPushToTalk();
});

pttBtnEl.addEventListener("pointercancel", () => {
  void endPushToTalk();
});

pttBtnEl.addEventListener("lostpointercapture", () => {
  void endPushToTalk();
});

videoBtnEl.addEventListener("click", async () => {
  const selected = await open({
    multiple: false,
    directory: false,
    filters: [{ name: "MP4 Video", extensions: ["mp4"] }],
  });
  if (selected === null) return;

  selectedVideoPath = selected;
  const fileName = selected.split(/[/\\]/).pop() ?? selected;
  videoNameEl.textContent = fileName;
  videoInfoEl.classList.add("visible");
  updateButtons(currentCoachState);
});

videoClearBtnEl.addEventListener("click", () => {
  selectedVideoPath = null;
  videoInfoEl.classList.remove("visible");
  videoNameEl.textContent = "";
  updateButtons(currentCoachState);
});

closeBtnEl.addEventListener("click", () => {
  getCurrentWindow().close();
});

minBtnEl.addEventListener("click", () => {
  getCurrentWindow().minimize();
});

/** Click-through toggle (Stretch Goal). */
const clickBtn = clickBtnEl;
clickBtn?.addEventListener("click", async () => {
  isClickThrough = !isClickThrough;
  await getCurrentWindow().setIgnoreCursorEvents(isClickThrough);
  clickBtn.classList.toggle("active", isClickThrough);
  clickBtn.title = isClickThrough
    ? "Click-through ON"
    : "Click-through OFF";
  statusTextEl.textContent = isClickThrough
    ? "Click-through ON"
    : "Connected";
});

window.addEventListener("keydown", (event) => {
  if (event.code !== "Space" || event.repeat) return;
  event.preventDefault();
  void startPushToTalk();
});

window.addEventListener("keyup", (event) => {
  if (event.code !== "Space") return;
  void endPushToTalk();
});

// ---------------------------------------------------------------------------
// Polling loop
// ---------------------------------------------------------------------------

window.setInterval(() => {
  void pollState();
}, POLL_INTERVAL_STATE);

// ---------------------------------------------------------------------------
// パーティクル球体の初期化
// ---------------------------------------------------------------------------

/**
 * パーティクル orb を初期化する。
 * 300 個の div を生成し、各パーティクルにランダムな球面座標の軌道アニメーションと
 * delay を割り当てる。SCSS の @for ループを JS に変換したもの。
 */
function initParticleOrb(): void {
  const TOTAL_PARTICLES = 300;
  const ORB_RADIUS_PX = 50;
  const ORBIT_DURATION_S = 14;

  const wrap = document.getElementById("particleWrap");
  if (!wrap) return;

  // 動的にキーフレームを挿入する stylesheet
  const styleSheet = document.createElement("style");
  document.head.appendChild(styleSheet);
  const sheet = styleSheet.sheet;
  if (!sheet) return;

  // 金アクセント用のインデックス（10% のパーティクルを金色にする）
  const GOLD_RATIO = 0.1;

  for (let i = 1; i <= TOTAL_PARTICLES; i++) {
    // ランダムな球面座標
    const zAngle = Math.random() * 360;
    const yAngle = Math.random() * 360;

    // パーティクル DOM 要素
    const el = document.createElement("div");
    el.className = "c";

    // 金アクセントパーティクル（idle 時に金のきらめきを出す）
    const isGold = Math.random() < GOLD_RATIO;
    if (isGold) {
      el.classList.add("c--gold");
    }

    // animation-delay でパーティクルの出現をずらす
    const delay = i * 0.01;

    // キーフレーム: パーティクルは球体表面に移動して留まる
    // 散らばらず、表面付近で微細な揺らぎのみ
    const kfName = `orbit${i}`;
    const wobble = (Math.random() - 0.5) * 6; // -3px ~ +3px の揺らぎ

    const keyframeRule = `
      @keyframes ${kfName} {
        0% {
          opacity: 0;
          transform: rotateZ(${-zAngle}deg) rotateY(${yAngle}deg)
                     translateX(0px) rotateZ(${zAngle}deg);
        }
        20% {
          opacity: var(--particle-opacity, 0.6);
          transform: rotateZ(${-zAngle}deg) rotateY(${yAngle}deg)
                     translateX(${ORB_RADIUS_PX}px) rotateZ(${zAngle}deg);
        }
        50% {
          opacity: var(--particle-opacity, 0.6);
          transform: rotateZ(${-zAngle}deg) rotateY(${yAngle}deg)
                     translateX(${ORB_RADIUS_PX + wobble}px) rotateZ(${zAngle}deg);
        }
        80% {
          opacity: var(--particle-opacity, 0.6);
          transform: rotateZ(${-zAngle}deg) rotateY(${yAngle}deg)
                     translateX(${ORB_RADIUS_PX}px) rotateZ(${zAngle}deg);
        }
        100% {
          opacity: var(--particle-opacity, 0.6);
          transform: rotateZ(${-zAngle}deg) rotateY(${yAngle}deg)
                     translateX(${ORB_RADIUS_PX}px) rotateZ(${zAngle}deg);
        }
      }
    `;

    sheet.insertRule(keyframeRule, sheet.cssRules.length);

    el.style.animation = `${kfName} ${ORBIT_DURATION_S}s infinite`;
    el.style.animationDelay = `${delay}s`;

    wrap.appendChild(el);
  }

  // 金アクセントパーティクルの追加スタイル
  sheet.insertRule(`
    .c--gold {
      background-color: hsla(42, 90%, 65%, 1) !important;
      width: 2.5px !important;
      height: 2.5px !important;
      box-shadow: 0 0 3px rgba(200, 155, 60, 0.6);
    }
  `, sheet.cssRules.length);

  // speaking 状態の金ハイライト: 金パーティクルをより明るく
  sheet.insertRule(`
    .orb--speaking .c--gold {
      background-color: hsla(45, 100%, 75%, 1) !important;
      box-shadow: 0 0 5px rgba(255, 215, 100, 0.8);
    }
  `, sheet.cssRules.length);
}

// DOM 準備完了後にパーティクルを初期化
initParticleOrb();
