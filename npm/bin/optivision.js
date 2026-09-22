#!/usr/bin/env node
/*
 * npx launcher for the `optivision` Python CLI.
 *
 * On first run it creates a private virtual environment in the user cache
 * directory and installs the PyPI release whose version matches this npm
 * package. Every later run starts the CLI directly. All arguments are passed
 * through unchanged.
 *
 * Environment variables:
 *   OPTIVISION_PYTHON     Python interpreter to use (default: first of py -3 / python3 / python)
 *   OPTIVISION_EXTRAS     pip extras to install, comma separated (default: "corpus"; "" for none)
 *                         e.g. OPTIVISION_EXTRAS=vlm,corpus for the real encoders
 *   OPTIVISION_PIP_SPEC   full pip requirement, overrides the version and extras
 *                         (a local wheel path works, for testing)
 *   OPTIVISION_HOME       where the environments are kept
 *   OPTIVISION_REINSTALL  set to 1 to rebuild the environment
 */
"use strict";

const { spawnSync, spawn } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const PKG = require("../package.json");
const MIN_PY = [3, 10];
const IS_WIN = process.platform === "win32";

function log(msg) {
  process.stderr.write(`optivision-rag: ${msg}\n`);
}

function fail(msg) {
  log(msg);
  process.exit(1);
}

function cacheRoot() {
  if (process.env.OPTIVISION_HOME) return process.env.OPTIVISION_HOME;
  if (IS_WIN) {
    const local = process.env.LOCALAPPDATA || path.join(os.homedir(), "AppData", "Local");
    return path.join(local, "optivision-rag");
  }
  if (process.platform === "darwin") {
    return path.join(os.homedir(), "Library", "Caches", "optivision-rag");
  }
  return path.join(process.env.XDG_CACHE_HOME || path.join(os.homedir(), ".cache"), "optivision-rag");
}

function splitCmd(cmd) {
  // "py -3" -> ["py", ["-3"]]; an existing path (even with spaces) stays whole.
  if (fs.existsSync(cmd)) return [cmd, []];
  const parts = cmd.trim().split(/\s+/);
  return [parts[0], parts.slice(1)];
}

function probePython(cmd) {
  const [exe, pre] = splitCmd(cmd);
  const r = spawnSync(exe, [...pre, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"], {
    encoding: "utf8",
    windowsHide: true,
  });
  if (r.status !== 0 || !r.stdout) return null;
  const [maj, min] = r.stdout.trim().split(".").map(Number);
  const ok = maj > MIN_PY[0] || (maj === MIN_PY[0] && min >= MIN_PY[1]);
  return { exe, pre, version: `${maj}.${min}`, tooOld: !ok };
}

function findPython() {
  const candidates = process.env.OPTIVISION_PYTHON
    ? [process.env.OPTIVISION_PYTHON]
    : IS_WIN
      ? ["py -3", "python", "python3"]
      : ["python3", "python"];
  const tooOld = [];
  for (const c of candidates) {
    const p = probePython(c);
    if (p && !p.tooOld) return p;
    if (p) tooOld.push(`${c} (${p.version})`);
  }
  const need = MIN_PY.join(".");
  if (tooOld.length) {
    fail(`Python ${need}+ is required; found only ${tooOld.join(", ")}. Set OPTIVISION_PYTHON to a newer interpreter.`);
  }
  fail(
    `Python ${need}+ is required but none was found on PATH. ` +
      "Install it from https://www.python.org/downloads/ or set OPTIVISION_PYTHON.",
  );
}

function requirement() {
  if (process.env.OPTIVISION_PIP_SPEC) return process.env.OPTIVISION_PIP_SPEC;
  const raw = process.env.OPTIVISION_EXTRAS === undefined ? "corpus" : process.env.OPTIVISION_EXTRAS;
  const extras = raw
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  return `optivision-rag${extras.length ? `[${extras.join(",")}]` : ""}==${PKG.version}`;
}

function envDirFor(spec) {
  const tag = spec.replace(/[^A-Za-z0-9.=_-]+/g, "_").slice(-60);
  return path.join(cacheRoot(), `env-${PKG.version}-${tag}`);
}

function venvBin(dir, name) {
  return IS_WIN ? path.join(dir, "Scripts", `${name}.exe`) : path.join(dir, "bin", name);
}

function runQuiet(exe, args) {
  // Installer output goes to stderr so the CLI's stdout stays clean for pipes.
  const r = spawnSync(exe, args, { stdio: ["ignore", process.stderr, process.stderr], windowsHide: true });
  if (r.error) fail(`could not run ${exe}: ${r.error.message}`);
  return r.status === 0;
}

function ensureInstalled() {
  const spec = requirement();
  const dir = envDirFor(spec);
  const marker = path.join(dir, ".optivision-installed");
  const cli = venvBin(dir, "optivision");

  const ready = fs.existsSync(marker) && fs.existsSync(cli);
  if (ready && process.env.OPTIVISION_REINSTALL !== "1") return cli;

  const py = findPython();
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(path.dirname(dir), { recursive: true });
  log(`first run: installing ${spec} with Python ${py.version} into ${dir}`);
  if (!runQuiet(py.exe, [...py.pre, "-m", "venv", dir])) {
    fs.rmSync(dir, { recursive: true, force: true });
    fail("could not create a virtual environment (on Debian/Ubuntu: apt install python3-venv)");
  }
  const venvPy = venvBin(dir, "python");
  const pipArgs = ["-m", "pip", "install", "--disable-pip-version-check", "--quiet", spec];
  if (!runQuiet(venvPy, pipArgs)) {
    fs.rmSync(dir, { recursive: true, force: true });
    fail(`pip install ${spec} failed (see the output above)`);
  }
  fs.writeFileSync(marker, `${spec}\n`);
  log("installed; later runs start immediately");
  return cli;
}

function main() {
  const cli = ensureInstalled();
  const child = spawn(cli, process.argv.slice(2), { stdio: "inherit" });
  for (const sig of ["SIGINT", "SIGTERM"]) {
    process.on(sig, () => child.kill(sig));
  }
  child.on("error", (e) => fail(`could not start ${cli}: ${e.message}`));
  child.on("exit", (code, signal) => {
    if (signal) process.kill(process.pid, signal);
    else process.exit(code === null ? 1 : code);
  });
}

main();
