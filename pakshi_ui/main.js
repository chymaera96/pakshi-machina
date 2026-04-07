const { app, BrowserWindow, ipcMain } = require("electron");
const path = require("path");
const { spawn } = require("child_process");
const readline = require("readline");

let mainWindow = null;
let worker = null;
let workerPaths = null;

function sendToWorker(payload) {
  if (!worker || worker.killed) {
    return;
  }
  worker.stdin.write(JSON.stringify(payload) + "\n");
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 820,
    backgroundColor: "#0f1115",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
    },
  });
  mainWindow.loadFile(path.join(__dirname, "index.html"));
}

function startWorker() {
  const repoRoot = path.resolve(__dirname, "..");
  const python = process.env.PAKSHI_PYTHON_PATH || path.join(repoRoot, ".venv", "bin", "python");
  const model = process.env.PAKSHI_MODEL_PATH || path.join(repoRoot, "models", "model.onnx");
  const bundle = process.env.PAKSHI_BUNDLE_PATH || path.join(repoRoot, "pakshi_bundle");
  const args = [path.join(repoRoot, "pakshi_worker.py"), "--model", model];
  if (process.env.PAKSHI_BUNDLE_PATH) {
    args.push("--bundle", bundle);
  }
  if (process.env.PAKSHI_ARM_ON_BOOT === "1") {
    args.push("--arm");
  }
  workerPaths = { python, model, bundle };

  worker = spawn(python, args, {
    cwd: repoRoot,
    stdio: ["pipe", "pipe", "pipe"],
  });

  worker.on("spawn", () => {
    if (mainWindow) {
      mainWindow.webContents.send("worker-event", {
        type: "ui_boot",
        ...workerPaths,
      });
    }
  });

  const rl = readline.createInterface({ input: worker.stdout });
  rl.on("line", (line) => {
    if (!mainWindow) {
      return;
    }
    try {
      mainWindow.webContents.send("worker-event", JSON.parse(line));
    } catch (error) {
      mainWindow.webContents.send("worker-event", { type: "error", message: String(error) });
    }
  });

  worker.stderr.on("data", (chunk) => {
    if (mainWindow) {
      mainWindow.webContents.send("worker-event", {
        type: "error",
        message: chunk.toString(),
      });
    }
  });

  worker.on("exit", (code, signal) => {
    if (mainWindow) {
      mainWindow.webContents.send("worker-event", {
        type: "worker_exit",
        code,
        signal,
      });
    }
  });
}

app.whenReady().then(() => {
  createWindow();
  startWorker();

  ipcMain.handle("worker-command", async (_, payload) => {
    sendToWorker(payload);
    return { ok: true };
  });

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", () => {
  if (worker && !worker.killed) {
    worker.kill();
  }
});
