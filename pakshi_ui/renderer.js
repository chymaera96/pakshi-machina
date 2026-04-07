const statePill = document.getElementById("state-pill");
const stateJson = document.getElementById("state-json");
const segmentsList = document.getElementById("segments-list");
const playbackList = document.getElementById("playback-list");
const bundleDir = document.getElementById("bundle-dir");
const bootInfo = document.getElementById("boot-info");

function appendListItem(list, text) {
  const item = document.createElement("li");
  item.textContent = text;
  list.prepend(item);
}

document.getElementById("load-corpus").addEventListener("click", () => {
  window.pakshi.sendCommand({
    command: "load_corpus",
    bundle_dir: bundleDir.value,
  });
});

document.getElementById("arm-worker").addEventListener("click", () => {
  window.pakshi.sendCommand({ command: "arm" });
});

document.getElementById("disarm-worker").addEventListener("click", () => {
  window.pakshi.sendCommand({ command: "disarm" });
});

document.getElementById("stop-all").addEventListener("click", () => {
  window.pakshi.sendCommand({ command: "stop_all" });
});

document.getElementById("refresh-state").addEventListener("click", () => {
  window.pakshi.sendCommand({ command: "get_state" });
});

window.pakshi.onWorkerEvent((event) => {
  if (event.type === "ui_boot") {
    bootInfo.textContent = `python: ${event.python} | model: ${event.model} | bundle: ${event.bundle}`;
  }
  if (event.type === "state") {
    statePill.textContent = event.state;
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
  if (event.type === "worker_exit") {
    appendListItem(playbackList, `worker exited code=${event.code} signal=${event.signal}`);
  }
  stateJson.textContent = JSON.stringify(event, null, 2);
});
