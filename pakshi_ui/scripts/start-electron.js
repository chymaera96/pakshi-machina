#!/usr/bin/env node
"use strict";

const { spawn } = require("child_process");
const path = require("path");

const electronBinary = require("electron");
const scriptArgs = process.argv.slice(2);
const trace = scriptArgs.includes("--trace");
const forwardedArgs = scriptArgs.filter((arg) => arg !== "--trace");

const env = { ...process.env };
delete env.ELECTRON_RUN_AS_NODE;
if (trace) {
  env.ELECTRON_ENABLE_LOGGING = env.ELECTRON_ENABLE_LOGGING || "1";
}

const child = spawn(electronBinary, forwardedArgs.length ? forwardedArgs : ["."], {
  cwd: path.resolve(__dirname, ".."),
  env,
  stdio: "inherit",
});

child.on("exit", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code ?? 0);
});

