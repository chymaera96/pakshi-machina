const statePill = document.getElementById("state-pill");
const stateJson = document.getElementById("state-json");
const segmentsList = document.getElementById("segments-list");
const playbackList = document.getElementById("playback-list");
const setupList = document.getElementById("setup-list");
const setupSummary = document.getElementById("setup-summary");
const setupBundleDir = document.getElementById("bundle-dir");
const performanceBundleDir = document.getElementById("bundle-dir-performance");
const setupModelFamily = document.getElementById("model-family");
const performanceModelFamily = document.getElementById("model-family-performance");
const bootInfo = document.getElementById("boot-info");
const cuePanel = document.getElementById("cue-panel");
const cueLabel = document.getElementById("cue-label");
const cueText = document.getElementById("cue-text");
const levelFill = document.getElementById("level-fill");
const monitorReadout = document.getElementById("monitor-readout");
const openMarker = document.getElementById("open-marker");
const closeMarker = document.getElementById("close-marker");
const pitchThresholdInput = document.getElementById("pitch-threshold-cents");
const pitchThresholdValue = document.getElementById("pitch-threshold-cents-value");
const pitchIgnoreShortGapsInput = document.getElementById("pitch-ignore-short-gaps");

let lastErrorMessage = null;
let isCalibrated = false;
let currentState = "idle";
let setupMode = true;
let gateOpenDb = -42;
let gateCloseDb = -48;
const meterFloorDb = -90;
let pitchControlsReady = false;
let runtimeBundleOverrideActive = false;
const MAX_LOG_ITEMS = 60;
const METER_UPDATE_INTERVAL_MS = 80;
const STATE_JSON_UPDATE_INTERVAL_MS = 150;
let latestHotMicEvent = null;
let meterFlushHandle = null;
let latestStateJsonEvent = null;
let stateJsonFlushHandle = null;

const STATE_COPY = {
  idle: ["Idle", "Enter setup or arm the mic for performance."],
  setup: ["Setup", "Capture room noise, then capture realistic singing."],
  listening: ["Listening", "Microphone hot. Waiting for a sung phrase."],
  in_phrase: ["Recording Phrase", "Phrase detected. Capturing live audio."],
  processing: ["Retrieving", "Embedding pitch-segmented windows with the selected retrieval backend and finding nearest neighbours."],
  playing_sequence: ["Playing Response", "Retrieved bird sounds are following the singer's pitch-segment timing."],
};

const ACTIVE_STATES = new Set(["listening", "in_phrase", "processing", "playing_sequence"]);

function appendListItem(list, text) {
  const item = document.createElement("li");
  item.textContent = text;
  list.prepend(item);
  while (list.children.length > MAX_LOG_ITEMS) {
    list.removeChild(list.lastChild);
  }
}

function flushMeterEvent() {
  meterFlushHandle = null;
  if (!latestHotMicEvent) {
    return;
  }
  const event = latestHotMicEvent;
  latestHotMicEvent = null;
  gateOpenDb = event.gate_open_db ?? gateOpenDb;
  gateCloseDb = event.gate_close_db ?? gateCloseDb;
  updateMarkers();
  const envelope = Math.max(0, Math.min(1, (event.envelope_db + 90) / 90));
  levelFill.style.width = `${envelope * 100}%`;
  monitorReadout.textContent = `level ${event.envelope_db.toFixed(1)} dB | gate ${event.gate_state}`;
}

function queueMeterEvent(event) {
  latestHotMicEvent = event;
  if (meterFlushHandle !== null) {
    return;
  }
  meterFlushHandle = window.setTimeout(flushMeterEvent, METER_UPDATE_INTERVAL_MS);
}

function flushStateJsonEvent() {
  stateJsonFlushHandle = null;
  if (!latestStateJsonEvent) {
    return;
  }
  stateJson.textContent = JSON.stringify(latestStateJsonEvent, null, 2);
  latestStateJsonEvent = null;
}

function updateStateJson(event, immediate = false) {
  if (immediate) {
    latestStateJsonEvent = null;
    if (stateJsonFlushHandle !== null) {
      window.clearTimeout(stateJsonFlushHandle);
      stateJsonFlushHandle = null;
    }
    stateJson.textContent = JSON.stringify(event, null, 2);
    return;
  }
  latestStateJsonEvent = event;
  if (stateJsonFlushHandle !== null) {
    return;
  }
  stateJsonFlushHandle = window.setTimeout(flushStateJsonEvent, STATE_JSON_UPDATE_INTERVAL_MS);
}

function pulseButton(button) {
  if (!button) {
    return;
  }
  button.classList.add("is-pressed");
  window.setTimeout(() => button.classList.remove("is-pressed"), 140);
}

function currentBundleDir() {
  return document.body.dataset.mode === "performance" ? performanceBundleDir.value.trim() : setupBundleDir.value.trim();
}

function syncBundleFields(value) {
  setupBundleDir.value = value;
  performanceBundleDir.value = value;
}

function syncModelFamilyFields(value) {
  setupModelFamily.value = value;
  performanceModelFamily.value = value;
}

function ensureCorpusLoaded() {
  const trimmedBundle = currentBundleDir();
  if (!trimmedBundle) {
    return;
  }
  syncBundleFields(trimmedBundle);
  window.pakshi.sendCommand({
    command: "load_corpus",
    bundle_dir: trimmedBundle,
  });
}

async function applyModelFamily(value) {
  syncModelFamilyFields(value);
  try {
    const info = await window.pakshi.setModelFamily(value);
    runtimeBundleOverrideActive = Boolean(info.bundleOverrideActive);
    if (!runtimeBundleOverrideActive && info.bundle) {
      syncBundleFields(info.bundle);
    }
    bootInfo.textContent =
      `python: ${info.python} | family: ${info.modelFamily} | model: ${info.model} | pitch: ${info.pitchModel} | bundle: ${info.bundle}`;
  } catch (error) {
    appendListItem(playbackList, `error: ${String(error.message || error)}`);
  }
}

function setMode(mode) {
  document.body.dataset.mode = mode;
}

function updateMarkers() {
  const normalize = (db) => `${Math.max(0, Math.min(100, ((db - meterFloorDb) / (0 - meterFloorDb)) * 100))}%`;
  openMarker.style.left = normalize(gateOpenDb);
  closeMarker.style.left = normalize(gateCloseDb);
}

function updatePitchThresholdReadout(value) {
  pitchThresholdValue.textContent = `${Math.round(Number(value))} cents`;
}

function syncPitchControls(config) {
  if (!config) {
    return;
  }
  pitchControlsReady = false;
  pitchThresholdInput.value = String(config.pitch_change_threshold_cents ?? 100);
  pitchIgnoreShortGapsInput.checked = Boolean(config.pitch_ignore_short_gaps ?? true);
  updatePitchThresholdReadout(pitchThresholdInput.value);
  pitchControlsReady = true;
}

function pushPitchParams() {
  if (!pitchControlsReady) {
    return;
  }
  window.pakshi.sendCommand({
    command: "set_params",
    params: {
      pitch_change_threshold_cents: Number(pitchThresholdInput.value),
      pitch_ignore_short_gaps: Boolean(pitchIgnoreShortGapsInput.checked),
    },
  });
}

function updateSetupSummary(event) {
  const noise = event.noise_floor_db;
  const soft = event.singing_soft_db;
  const open = event.gate_open_db;
  const close = event.gate_close_db;
  setupSummary.textContent =
    `noise ${noise == null ? "--" : `${noise.toFixed(1)} dB`} | ` +
    `soft ${soft == null ? "--" : `${soft.toFixed(1)} dB`} | ` +
    `open ${open == null ? "--" : `${open.toFixed(1)} dB`} | ` +
    `close ${close == null ? "--" : `${close.toFixed(1)} dB`}`;
}

function setCueState(state) {
  currentState = state;
  cuePanel.dataset.state = state;
  const [label, text] = STATE_COPY[state] || [state, "Worker status updated."];
  cueLabel.textContent = label;
  cueText.textContent = text;
  document.getElementById("arm-worker").classList.toggle("is-active", ACTIVE_STATES.has(state));
}

function enterSetupMode() {
  setMode("setup");
  ensureCorpusLoaded();
  window.pakshi.sendCommand({ command: "start_setup" });
}

document.getElementById("setup-enter").addEventListener("click", (event) => {
  pulseButton(event.currentTarget);
  enterSetupMode();
});

document.getElementById("capture-noise").addEventListener("click", (event) => {
  pulseButton(event.currentTarget);
  ensureCorpusLoaded();
  if (!setupMode) {
    window.pakshi.sendCommand({ command: "start_setup" });
  }
  window.pakshi.sendCommand({ command: "capture_noise_floor" });
});

document.getElementById("capture-singing").addEventListener("click", (event) => {
  pulseButton(event.currentTarget);
  ensureCorpusLoaded();
  if (!setupMode) {
    window.pakshi.sendCommand({ command: "start_setup" });
  }
  window.pakshi.sendCommand({ command: "capture_singing_level" });
});

document.getElementById("reset-setup").addEventListener("click", (event) => {
  pulseButton(event.currentTarget);
  window.pakshi.sendCommand({ command: "reset_setup" });
});

document.getElementById("reopen-setup").addEventListener("click", (event) => {
  pulseButton(event.currentTarget);
  enterSetupMode();
});

document.getElementById("arm-worker").addEventListener("click", (event) => {
  pulseButton(event.currentTarget);
  ensureCorpusLoaded();
  if (ACTIVE_STATES.has(currentState)) {
    window.pakshi.sendCommand({ command: "disarm" });
    return;
  }
  window.pakshi.sendCommand({ command: "arm" });
});

setupBundleDir.addEventListener("input", (event) => syncBundleFields(event.target.value));
performanceBundleDir.addEventListener("input", (event) => syncBundleFields(event.target.value));
setupModelFamily.addEventListener("change", (event) => applyModelFamily(event.target.value));
performanceModelFamily.addEventListener("change", (event) => applyModelFamily(event.target.value));
pitchThresholdInput.addEventListener("input", (event) => {
  updatePitchThresholdReadout(event.target.value);
  pushPitchParams();
});
pitchIgnoreShortGapsInput.addEventListener("change", pushPitchParams);

window.addEventListener("load", () => {
  window.pakshi.sendCommand({ command: "get_state" });
});

window.pakshi.onWorkerEvent((event) => {
  if (event.type === "ui_boot") {
    runtimeBundleOverrideActive = Boolean(event.bundleOverrideActive);
    bootInfo.textContent = `python: ${event.python} | family: ${event.modelFamily} | model: ${event.model} | pitch: ${event.pitchModel} | bundle: ${event.bundle}`;
    syncBundleFields(event.bundle);
    syncModelFamilyFields(event.modelFamily);
  }
  if (event.type === "error" || event.type === "setup_error") {
    const message = String(event.message || "").trim();
    lastErrorMessage = message || lastErrorMessage;
    appendListItem(playbackList, `error: ${message}`);
    updateStateJson(event, true);
    if (event.type === "setup_error") {
      cueLabel.textContent = "Setup Error";
      cueText.textContent = message;
    }
    return;
  }
  if (event.type === "state") {
    statePill.textContent = event.state;
    isCalibrated = Boolean(event.calibrated);
    setupMode = Boolean(event.setup_mode);
    gateOpenDb = event.gate_open_db ?? gateOpenDb;
    gateCloseDb = event.gate_close_db ?? gateCloseDb;
    updateMarkers();
    updateSetupSummary(event);
    syncPitchControls(event.config);
    if (event.model_family) {
      syncModelFamilyFields(event.model_family);
    }
    setMode(setupMode || !isCalibrated ? "setup" : "performance");
    setCueState(event.state);
  }
  if (event.type === "setup_started") {
    appendListItem(setupList, "setup mode entered");
    cueLabel.textContent = "Setup";
    cueText.textContent = "Capture room noise first.";
  }
  if (event.type === "noise_floor_capture_started") {
    appendListItem(setupList, `capturing noise floor for ${event.duration_seconds.toFixed(1)} s`);
    cueLabel.textContent = "Capture Noise";
    cueText.textContent = "Stay silent while the room noise floor is measured.";
  }
  if (event.type === "noise_floor_captured") {
    appendListItem(setupList, `noise floor ${event.noise_floor_db.toFixed(1)} dB`);
    cueLabel.textContent = "Noise Captured";
    cueText.textContent = "Now capture realistic singing for 5-10 seconds.";
  }
  if (event.type === "singing_level_capture_started") {
    appendListItem(setupList, `capturing singing level for ${event.duration_seconds.toFixed(1)} s`);
    cueLabel.textContent = "Capture Singing";
    cueText.textContent = "Sing with realistic dynamics, including softer entries.";
  }
  if (event.type === "singing_level_captured") {
    gateOpenDb = event.gate_open_db;
    gateCloseDb = event.gate_close_db;
    updateMarkers();
    appendListItem(
      setupList,
      `singing soft ${event.singing_soft_db.toFixed(1)} dB | open ${event.gate_open_db.toFixed(1)} dB | close ${event.gate_close_db.toFixed(1)} dB`
    );
    updateSetupSummary(event);
    cueLabel.textContent = "Gate Derived";
    cueText.textContent = "Setup complete. Use the record control for performance.";
  }
  if (event.type === "setup_ready") {
    gateOpenDb = event.gate_open_db;
    gateCloseDb = event.gate_close_db;
    updateMarkers();
    updateSetupSummary(event);
    appendListItem(setupList, `setup ready: open ${event.gate_open_db.toFixed(1)} dB | close ${event.gate_close_db.toFixed(1)} dB`);
    cueLabel.textContent = "Ready";
    cueText.textContent = "Setup complete. Use the record control for performance.";
  }
  if (event.type === "setup_reset") {
    appendListItem(setupList, "setup reset");
    setupSummary.textContent = "noise -- | soft -- | open -- | close --";
  }
  if (event.type === "mic_status" && !event.active) {
    levelFill.style.width = "0%";
    monitorReadout.textContent = "level -inf dB | gate below";
  }
  if (event.type === "mic_level") {
    queueMeterEvent(event);
    updateStateJson(event, false);
    return;
  }
  if (event.type === "phrase_summary") {
    const starts = (event.segment_start_offsets_seconds || []).map((value) => value.toFixed(2)).join(", ");
    appendListItem(
      playbackList,
      `phrase ${event.phrase_id}: ${event.duration_seconds.toFixed(2)} s, ${event.num_segments} pitch segments [${starts}]`
    );
  }
  if (event.type === "segments_created") {
    const starts = (event.segment_start_offsets_seconds || []).map((value) => value.toFixed(2)).join(", ");
    appendListItem(segmentsList, `phrase ${event.phrase_id}: ${event.num_segments} segments [${starts}]`);
  }
  if (event.type === "pitch_debug_saved") {
    appendListItem(playbackList, `phrase ${event.phrase_id}: pitch debug saved (${event.num_segments} segments) -> ${event.path}`);
  }
  if (event.type === "retrieval_sequence_ready") {
    appendListItem(
      segmentsList,
      `phrase ${event.phrase_id}: matches ${event.matches.map((match) => `${match.metadata.name || match.metadata.path || match.corpus_index}@${match.scheduled_offset_seconds.toFixed(2)}s`).join(" | ")}`
    );
    appendListItem(playbackList, `phrase ${event.phrase_id}: scheduling ${event.num_segments} vocalisations from ${event.num_segments} pitch segments`);
  }
  if (event.type === "segment_playback_started") {
    appendListItem(playbackList, `phrase ${event.phrase_id} seg ${event.segment_index} started @ ${event.scheduled_offset_seconds.toFixed(2)} s`);
  }
  if (event.type === "segment_playback_finished") {
    appendListItem(playbackList, `phrase ${event.phrase_id} seg ${event.segment_index} finished`);
  }
  if (event.type === "sequence_playback_finished") {
    appendListItem(playbackList, `phrase ${event.phrase_id}: sequence finished`);
  }
  if (event.type === "queue_cleared") {
    appendListItem(playbackList, "queue cleared, phrase buffer reset");
  }
  if (event.type === "worker_exit") {
    appendListItem(playbackList, `worker exited code=${event.code} signal=${event.signal}`);
    if (event.stderrTail) {
      appendListItem(playbackList, `stderr: ${String(event.stderrTail).trim()}`);
    }
  }
  updateStateJson(event, true);
});
