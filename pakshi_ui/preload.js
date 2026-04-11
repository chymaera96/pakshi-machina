const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("pakshi", {
  sendCommand(payload) {
    return ipcRenderer.invoke("worker-command", payload);
  },
  setModelFamily(modelFamily) {
    return ipcRenderer.invoke("set-model-family", modelFamily);
  },
  onWorkerEvent(handler) {
    ipcRenderer.on("worker-event", (_, payload) => handler(payload));
  },
});
