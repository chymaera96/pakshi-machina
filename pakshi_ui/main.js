const { app, BrowserWindow, ipcMain } = require("electron");
const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");
const readline = require("readline");

let mainWindow = null;
let worker = null;
let workerPaths = null;
let workerStderrTail = "";

function emitWorkerEvent(payload) {
  if (!mainWindow || mainWindow.isDestroyed()) {
    return;
  }
  if (!mainWindow.webContents || mainWindow.webContents.isDestroyed()) {
    return;
  }
  mainWindow.webContents.send("worker-event", payload);
}

function resolvePython(repoRoot) {
  if (process.env.PAKSHI_PYTHON_PATH) {
    return process.env.PAKSHI_PYTHON_PATH;
  }
  if (process.env.CONDA_PREFIX) {
    return path.join(process.env.CONDA_PREFIX, "bin", "python");
  }
  return "python3";
}

function resolveModel(repoRoot) {
  if (process.env.PAKSHI_MODEL_PATH) {
    if (fs.existsSync(process.env.PAKSHI_MODEL_PATH)) {
      return process.env.PAKSHI_MODEL_PATH;
    }
  }
  const onnxFiles = fs
    .readdirSync(repoRoot)
    .filter((name) => name.toLowerCase().endsWith(".onnx"))
    .sort();
  if (onnxFiles.length === 1) {
    return path.join(repoRoot, onnxFiles[0]);
  }
  if (onnxFiles.length > 1) {
    throw new Error(
      `Multiple ONNX files found in repo root: ${onnxFiles.join(", ")}. Set PAKSHI_MODEL_PATH explicitly.`
    );
  }
  throw new Error("No ONNX file found in repo root. Add one or set PAKSHI_MODEL_PATH explicitly.");
}

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
  const python = resolvePython(repoRoot);
  const model = resolveModel(repoRoot);
  const bundle = process.env.PAKSHI_BUNDLE_PATH || path.join(repoRoot, "pakshi_bundle");
  const args = [path.join(repoRoot, "pakshi_worker.py"), "--model", model];
  args.push("--bundle", bundle);
  workerPaths = { python, model, bundle };
  workerStderrTail = "";

  worker = spawn(python, args, {
    cwd: repoRoot,
    env: {
      ...process.env,
      KMP_DUPLICATE_LIB_OK: process.env.KMP_DUPLICATE_LIB_OK || "TRUE",
    },
    stdio: ["pipe", "pipe", "pipe"],
  });

  worker.on("spawn", () => {
    emitWorkerEvent({
      type: "ui_boot",
      ...workerPaths,
    });
  });

  worker.on("error", (error) => {
    emitWorkerEvent({
      type: "error",
      message: `worker failed to start: ${String(error.message || error)}`,
    });
  });

  const rl = readline.createInterface({ input: worker.stdout });
  rl.on("line", (line) => {
    try {
      emitWorkerEvent(JSON.parse(line));
    } catch (error) {
      emitWorkerEvent({ type: "error", message: String(error) });
    }
  });

  worker.stderr.on("data", (chunk) => {
    const text = chunk.toString();
    workerStderrTail = `${workerStderrTail}${text}`.slice(-4000);
    emitWorkerEvent({
      type: "error",
      message: text,
    });
  });

  worker.on("exit", (code, signal) => {
    emitWorkerEvent({
      type: "worker_exit",
      code,
      signal,
      stderrTail: workerStderrTail || null,
    });
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
