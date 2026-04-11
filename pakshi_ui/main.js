const { app, BrowserWindow, ipcMain } = require("electron");
const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");
const readline = require("readline");

const DEFAULT_MODEL_FAMILY = "effnet_bio";
const MODEL_FAMILIES = ["effnet_bio", "crepe_latent"];
const BUNDLE_DIR_BY_FAMILY = {
  effnet_bio: "pakshi_bundle_effnet_bio",
  crepe_latent: "pakshi_bundle_crepe_latent",
};

let mainWindow = null;
let worker = null;
let workerPaths = null;
let workerStderrTail = "";
let selectedModelFamily = process.env.PAKSHI_MODEL_FAMILY || DEFAULT_MODEL_FAMILY;

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

function inferModelFamilyFromPath(modelPath) {
  const name = path.basename(modelPath).toLowerCase();
  if (name.includes("effnet")) {
    return "effnet_bio";
  }
  if (name.includes("crepe")) {
    return "crepe_latent";
  }
  throw new Error(`Could not infer model family from ${modelPath}. Set PAKSHI_MODEL_FAMILY explicitly.`);
}

function listAvailableFamilies(repoRoot) {
  return MODEL_FAMILIES.filter((family) => {
    try {
      resolveModel(repoRoot, family);
      return true;
    } catch (_error) {
      return false;
    }
  });
}

function resolveModel(repoRoot, modelFamily) {
  if (process.env.PAKSHI_MODEL_PATH && fs.existsSync(process.env.PAKSHI_MODEL_PATH)) {
    return process.env.PAKSHI_MODEL_PATH;
  }

  const candidates =
    modelFamily === "crepe_latent"
      ? [
          path.join(repoRoot, "crepe_latent.onnx"),
          path.join(process.env.HOME || "", "Downloads", "trained_models", "crepe_latent.onnx"),
          path.join(process.env.HOME || "", "Downloads", "trained_models", "crepe", "crepe_latent.onnx"),
        ]
      : [
          path.join(repoRoot, "effnet_bio_zf_emb1024.onnx"),
          path.join(process.env.HOME || "", "Downloads", "trained_models", "effnet_bio", "effnet_bio_zf_emb1024.onnx"),
        ];

  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }

  const familyToken = modelFamily === "crepe_latent" ? "crepe" : "effnet";
  const onnxFiles = fs
    .readdirSync(repoRoot)
    .filter((name) => name.toLowerCase().endsWith(".onnx") && name.toLowerCase().includes(familyToken))
    .sort()
    .map((name) => path.join(repoRoot, name));
  const unique = [...new Set(onnxFiles)];
  if (unique.length === 1) {
    return unique[0];
  }
  if (unique.length > 1) {
    throw new Error(`Multiple ${modelFamily} ONNX files found: ${unique.join(", ")}. Set PAKSHI_MODEL_PATH explicitly.`);
  }
  throw new Error(`No ${modelFamily} ONNX found. Add one or set PAKSHI_MODEL_PATH explicitly.`);
}

function resolvePitchModel(repoRoot) {
  if (process.env.PAKSHI_PITCH_MODEL_PATH && fs.existsSync(process.env.PAKSHI_PITCH_MODEL_PATH)) {
    return process.env.PAKSHI_PITCH_MODEL_PATH;
  }
  const candidates = [
    path.join(repoRoot, "crepe_pitch.onnx"),
    path.join(repoRoot, "crepe_pitch_frames.onnx"),
    path.join(process.env.HOME || "", "Downloads", "trained_models", "crepe_pitch.onnx"),
    path.join(process.env.HOME || "", "Downloads", "trained_models", "crepe_pitch_frames.onnx"),
  ];
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  throw new Error("No CREPE pitch ONNX found. Add crepe_pitch.onnx to the repo root or set PAKSHI_PITCH_MODEL_PATH explicitly.");
}

function resolveBundle(repoRoot, modelFamily) {
  if (process.env.PAKSHI_BUNDLE_PATH) {
    return process.env.PAKSHI_BUNDLE_PATH;
  }
  return path.join(repoRoot, BUNDLE_DIR_BY_FAMILY[modelFamily] || BUNDLE_DIR_BY_FAMILY[DEFAULT_MODEL_FAMILY]);
}

function sendToWorker(payload) {
  if (!worker || worker.killed) {
    emitWorkerEvent({
      type: "error",
      message: "worker is not running; check startup errors and ensure the selected model files exist",
    });
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

function stopWorker() {
  if (worker && !worker.killed) {
    worker.kill();
  }
  worker = null;
}

function startWorker(overrides = {}) {
  const repoRoot = path.resolve(__dirname, "..");
  const python = resolvePython(repoRoot);

  let modelFamily = overrides.modelFamily || selectedModelFamily || DEFAULT_MODEL_FAMILY;
  if (process.env.PAKSHI_MODEL_PATH && !process.env.PAKSHI_MODEL_FAMILY) {
    modelFamily = inferModelFamilyFromPath(process.env.PAKSHI_MODEL_PATH);
  }
  if (!MODEL_FAMILIES.includes(modelFamily)) {
    throw new Error(`Unsupported model family '${modelFamily}'. Expected one of ${MODEL_FAMILIES.join(", ")}.`);
  }

  const model = overrides.model || resolveModel(repoRoot, modelFamily);
  const pitchModel = resolvePitchModel(repoRoot);
  const bundle = overrides.bundle || resolveBundle(repoRoot, modelFamily);
  const args = [
    path.join(repoRoot, "pakshi_worker.py"),
    "--model",
    model,
    "--model-family",
    modelFamily,
    "--pitch-model",
    pitchModel,
    "--bundle",
    bundle,
  ];

  selectedModelFamily = modelFamily;
  workerPaths = {
    python,
    model,
    modelFamily,
    pitchModel,
    bundle,
    availableFamilies: listAvailableFamilies(repoRoot),
    bundleOverrideActive: Boolean(process.env.PAKSHI_BUNDLE_PATH),
  };
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

  return workerPaths;
}

app.whenReady().then(() => {
  createWindow();
  ipcMain.handle("worker-command", async (_, payload) => {
    sendToWorker(payload);
    return { ok: true };
  });
  ipcMain.handle("set-model-family", async (_, modelFamily) => {
    const repoRoot = path.resolve(__dirname, "..");
    const resolvedModelFamily = modelFamily || selectedModelFamily || DEFAULT_MODEL_FAMILY;
    const model = resolveModel(repoRoot, resolvedModelFamily);
    const bundle = resolveBundle(repoRoot, resolvedModelFamily);
    const info = {
      python: workerPaths?.python || resolvePython(repoRoot),
      model,
      modelFamily: resolvedModelFamily,
      pitchModel: workerPaths?.pitchModel || resolvePitchModel(repoRoot),
      bundle,
      availableFamilies: listAvailableFamilies(repoRoot),
      bundleOverrideActive: Boolean(process.env.PAKSHI_BUNDLE_PATH),
    };

    if (!worker || worker.killed) {
      stopWorker();
      startWorker({ modelFamily: resolvedModelFamily, model, bundle });
      return { ok: true, ...info };
    }

    selectedModelFamily = resolvedModelFamily;
    sendToWorker({
      command: "set_model_backend",
      model_path: model,
      model_family: resolvedModelFamily,
      bundle_dir: bundle,
    });
    workerPaths = info;
    emitWorkerEvent({ type: "ui_boot", ...info });
    return { ok: true, ...info };
  });
  try {
    startWorker();
  } catch (error) {
    emitWorkerEvent({
      type: "error",
      message: String(error.message || error),
    });
  }

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
  stopWorker();
});
