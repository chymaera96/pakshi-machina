const statePill = document.getElementById("state-pill");
const stateJson = document.getElementById("state-json");
const segmentsList = document.getElementById("segments-list");
const playbackList = document.getElementById("playback-list");
const bundleDir = document.getElementById("bundle-dir");
const bootInfo = document.getElementById("boot-info");
const cuePanel = document.getElementById("cue-panel");
const cueLabel = document.getElementById("cue-label");
const cueText = document.getElementById("cue-text");
const levelFill = document.getElementById("level-fill");
const monitorReadout = document.getElementById("monitor-readout");
let lastErrorMessage = null;
let isCalibrated = false;
let currentState = "idle";

const STATE_COPY = {
  idle: ["Idle", "Tap the mic to calibrate or restart listening."],
  listening: ["Listening", "Microphone hot. Waiting for a sung phrase."],
  in_phrase: ["Recording Phrase", "Phrase detected. Capturing live audio."],
  processing: ["Retrieving", "Embedding segments and finding nearest neighbours."],
  playing_sequence: ["Playing Response", "Retrieved sequence is being played back."],
  calibrating: ["Calibrating VAD", "Sing three short solfege phrases with clear pauses."],
};

const ACTIVE_STATES = new Set(["calibrating", "listening", "in_phrase", "processing", "playing_sequence"]);

function appendListItem(list, text) {
  const item = document.createElement("li");
  item.textContent = text;
  list.prepend(item);
}

function pulseButton(button) {
  button.classList.add("is-pressed");
  window.setTimeout(() => button.classList.remove("is-pressed"), 140);
}

const primaryButton = document.getElementById("arm-worker");

primaryButton.addEventListener("click", () => {
  pulseButton(primaryButton);
  const trimmedBundle = bundleDir.value.trim();
  if (trimmedBundle && !ACTIVE_STATES.has(currentState)) {
    window.pakshi.sendCommand({
      command: "load_corpus",
      bundle_dir: trimmedBundle,
    });
  }
  if (ACTIVE_STATES.has(currentState)) {
    window.pakshi.sendCommand({ command: "disarm" });
    return;
  }
  if (!isCalibrated) {
    window.pakshi.sendCommand({ command: "start_calibration" });
  } else {
    window.pakshi.sendCommand({ command: "arm" });
  }
});

window.addEventListener("load", () => {
  window.pakshi.sendCommand({ command: "get_state" });
});

function setCueState(state) {
  currentState = state;
  cuePanel.dataset.state = state;
  const [label, text] = STATE_COPY[state] || [state, "Worker status updated."];
  cueLabel.textContent = label;
  cueText.textContent = text;
  primaryButton.classList.toggle("is-active", ACTIVE_STATES.has(state));
}

window.pakshi.onWorkerEvent((event) => {
  if (event.type === "ui_boot") {
    bootInfo.textContent = `python: ${event.python} | model: ${event.model} | bundle: ${event.bundle}`;
  }
  if (event.type === "error") {
    const message = String(event.message || "").trim();
    lastErrorMessage = message || lastErrorMessage;
    appendListItem(playbackList, `error: ${message}`);
    stateJson.textContent = JSON.stringify(event, null, 2);
    return;
  }
  if (event.type === "state") {
    statePill.textContent = event.state;
    isCalibrated = Boolean(event.calibrated);
    setCueState(event.state);
  }
  if (event.type === "calibration_progress") {
    cueLabel.textContent = "Calibrating VAD";
    cueText.textContent = `Detected ${event.phrase_count} / ${event.target_phrases} calibration phrases.`;
  }
  if (event.type === "calibration_complete") {
    isCalibrated = true;
    appendListItem(
      playbackList,
      `calibration complete: onset ${event.onset_threshold.toFixed(2)}, sustain ${event.sustain_threshold.toFixed(2)}`
    );
  }
  if (event.type === "mic_status" && !event.active) {
    levelFill.style.width = "0%";
    monitorReadout.textContent = "mic 0.00 | confidence 0.00";
  }
  if (event.type === "mic_level") {
    const level = Math.max(0, Math.min(1, event.level * 12));
    const confidence = Math.max(0, Math.min(1, event.confidence));
    levelFill.style.width = `${Math.max(level, confidence) * 100}%`;
    monitorReadout.textContent = `mic ${event.level.toFixed(3)} | confidence ${event.confidence.toFixed(3)}`;
  }
  if (event.type === "segments_created") {
    appendListItem(
      segmentsList,
      `phrase ${event.phrase_id}: ${event.num_segments} segments`
    );
  }
  if (event.type === "retrieval_sequence_ready") {
    appendListItem(
      segmentsList,
      `phrase ${event.phrase_id}: matches ${event.matches
        .map((match) => match.metadata.name || match.metadata.path || match.corpus_index)
        .join(" | ")}`
    );
    appendListItem(
      playbackList,
      `phrase ${event.phrase_id}: stitching ${event.num_segments} vocalisations`
    );
  }
  if (event.type === "segment_playback_started") {
    appendListItem(
      playbackList,
      `phrase ${event.phrase_id} seg ${event.segment_index} started`
    );
  }
  if (event.type === "segment_playback_finished") {
    appendListItem(
      playbackList,
      `phrase ${event.phrase_id} seg ${event.segment_index} finished`
    );
  }
  if (event.type === "queue_cleared") {
    appendListItem(playbackList, "queue cleared, phrase buffer reset");
  }
  if (event.type === "worker_exit") {
    appendListItem(playbackList, `worker exited code=${event.code} signal=${event.signal}`);
    if (event.stderrTail) {
      appendListItem(playbackList, `stderr: ${String(event.stderrTail).trim()}`);
      stateJson.textContent = JSON.stringify(
        {
          ...event,
          stderrTail: String(event.stderrTail).trim(),
        },
        null,
        2
      );
      return;
    }
    if (lastErrorMessage) {
      stateJson.textContent = JSON.stringify(
        {
          ...event,
          lastErrorMessage,
        },
        null,
        2
      );
      return;
    }
  }
  stateJson.textContent = JSON.stringify(event, null, 2);
});
