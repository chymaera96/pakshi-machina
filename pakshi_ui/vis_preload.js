const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("pakshiVis", {
  onEvent(handler) {
    ipcRenderer.on("vis-event", (_, payload) => handler(payload));
  },
});
