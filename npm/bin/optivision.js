#!/usr/bin/env node
/*
 * npx launcher for the `optivision` Python CLI.
 *
 * On first run it creates a private virtual environment in the user cache
 * directory and installs the PyPI release whose version matches this npm
 * package. Every later run starts the CLI directly. All arguments are passed
 * through unchanged.
 *
 * Easy mode (optivision.easy). `options` and `doctor` run in that same light
 * environment. `build`, `find`, `check` and `quickstart` need a model library
 * (PyTorch and a loader), so they run in a second, heavy environment that is
 * keyed by the pip extra: `easy-st` (sentence-transformers) or `easy-colpali`
 * (colpali-engine), chosen from the model's loader. The model comes from
 * --model, or for `find` and `check` from the index folder's manifest. Before
 * the heavy install the launcher prints the download size and, on Windows, that
 * pip installs the CPU-only build of PyTorch; the install output is not hidden.
 *
 * Environment variables:
 *   OPTIVISION_PYTHON     Python interpreter to use (default: first of py -3 / python3 / python)
 *   OPTIVISION_EXTRAS     pip extras to install, comma separated (default: "corpus"; "" for none)
 *                         e.g. OPTIVISION_EXTRAS=vlm,corpus for the real encoders
 *                         (the heavy environment adds easy-st or easy-colpali to whatever is set)
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

// Commands that load a model, and so need the heavy environment.
const HEAVY_COMMANDS = new Set(["build", "find", "check", "quickstart"]);
// Options of those commands that take a value (so the value is not read as the index or folder).
const VALUE_OPTIONS = new Set([
  "--model", "--target", "--path", "--work-dir", "--sample", "--queries", "--answers", "--max-disk", "-k", "--top-k",
]);
const EXTRA_ST = "easy-st";
const EXTRA_COLPALI = "easy-colpali";

// What a heavy install downloads, read from the PyPI file listings in October 2026: the torch 2.14.1
// wheel is 555 MB on Linux x86_64 (plus about 3 GB of NVIDIA CUDA libraries it depends on), 124 MB on
// Windows (the CPU build) and 127 MB on macOS arm64; the other libraries add roughly 0.2 GB.
const DOWNLOAD_SIZE = {
  linux: "roughly 3 to 4 GB (PyTorch with its CUDA libraries)",
  win32: "roughly 0.3 to 0.5 GB (the CPU build of PyTorch)",
  darwin: "roughly 0.3 to 0.5 GB",
};

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

// The extras named in OPTIVISION_EXTRAS: `undefined` when it is not set.
function configuredExtras() {
  if (process.env.OPTIVISION_EXTRAS === undefined) return undefined;
  return process.env.OPTIVISION_EXTRAS.split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

// The light environment's requirement: unchanged since 0.5 (default extra "corpus").
function requirement() {
  if (process.env.OPTIVISION_PIP_SPEC) return process.env.OPTIVISION_PIP_SPEC;
  const extras = configuredExtras() || ["corpus"];
  return `optivision-rag${extras.length ? `[${extras.join(",")}]` : ""}==${PKG.version}`;
}

// `spec` with `extra` added to its extras: "name==1" -> "name[extra]==1", "name[a]==1" -> "name[a,extra]==1",
// "name[a] @ file:///x.whl" -> "name[a,extra] @ file:///x.whl", "dist/x.whl" -> "dist/x.whl[extra]".
function withExtra(spec, extra) {
  const bracket = /^(.*?)\[([^\]]*)\](.*)$/.exec(spec);
  if (bracket) {
    const have = bracket[2].split(",").map((s) => s.trim()).filter(Boolean);
    if (!have.includes(extra)) have.push(extra);
    return `${bracket[1]}[${have.join(",")}]${bracket[3]}`;
  }
  const at = spec.indexOf(" @ ");
  if (at >= 0) return `${spec.slice(0, at)}[${extra}]${spec.slice(at)}`;
  const eq = spec.indexOf("==");
  if (eq >= 0) return `${spec.slice(0, eq)}[${extra}]${spec.slice(eq)}`;
  return `${spec}[${extra}]`;
}

// The heavy environment's requirement: the light one plus the loader's extra. Without OPTIVISION_EXTRAS
// nothing else is added (no "corpus"), so the environment holds only what the easy-mode commands need.
function heavyRequirement(extra) {
  if (process.env.OPTIVISION_PIP_SPEC) return withExtra(process.env.OPTIVISION_PIP_SPEC, extra);
  const extras = [...(configuredExtras() || [])];
  if (!extras.includes(extra)) extras.push(extra);
  return `optivision-rag[${extras.join(",")}]==${PKG.version}`;
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

// ------------------------------------------------------------------ routing (easy mode)

// The command: the first argument that is not an option.
function commandOf(args) {
  return args.find((a) => !a.startsWith("-"));
}

// The value of `--name X` or `--name=X`, or undefined.
function optionValue(args, name) {
  for (let i = 0; i < args.length; i++) {
    if (args[i] === name) return args[i + 1];
    if (args[i].startsWith(`${name}=`)) return args[i].slice(name.length + 1);
  }
  return undefined;
}

// Arguments after the command that are not options or option values (the folder, the index, the query).
function positionals(args) {
  const out = [];
  let seenCommand = false;
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a.startsWith("-")) {
      if (!a.includes("=") && VALUE_OPTIONS.has(a)) i++;
      continue;
    }
    if (!seenCommand) seenCommand = true;
    else out.push(a);
  }
  return out;
}

// The first string value stored under `key` anywhere in a parsed JSON document (a few levels deep).
function findKey(obj, key, depth = 0) {
  if (!obj || typeof obj !== "object" || depth > 4) return undefined;
  if (typeof obj[key] === "string") return obj[key];
  for (const value of Object.values(obj)) {
    const found = findKey(value, key, depth + 1);
    if (found !== undefined) return found;
  }
  return undefined;
}

// What an index folder says about its model: {loader, model}, either may be undefined. `my_pdfs` also
// finds `my_pdfs.ovx`. The manifest is read first, then the build record.
function indexInfo(indexPath) {
  if (!indexPath) return {};
  for (const dir of [indexPath, `${indexPath}.ovx`]) {
    for (const file of ["manifest.json", "build.json", path.join("state", "build.json")]) {
      try {
        const doc = JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
        const info = { loader: findKey(doc, "loader"), model: findKey(doc, "model") || findKey(doc, "model_id") };
        if (info.loader || info.model) return info;
      } catch (e) {
        // no such file, or not JSON: try the next one
      }
    }
  }
  return {};
}

// `colpali-engine:*` loaders need the colpali extra; every other loader (sentence-transformers, the
// remote-code recipes) loads through transformers, which the sentence-transformers extra installs.
function extraForLoader(loader) {
  return typeof loader === "string" && loader.startsWith("colpali-engine") ? EXTRA_COLPALI : EXTRA_ST;
}

// Where `args` should run: {heavy: false} or {heavy: true, extra, model, loader}. `lookupLoader(model)`
// returns a model's loader name (or null); it is only called when the loader is not already known.
function route(args, lookupLoader) {
  const command = commandOf(args);
  if (!HEAVY_COMMANDS.has(command) || args.includes("--help")) return { heavy: false, command };
  let model;
  let loader;
  if (command === "find" || command === "check") {
    ({ loader, model } = indexInfo(positionals(args)[0]));
  } else {
    model = optionValue(args, "--model");
  }
  if (!loader && model && lookupLoader) loader = lookupLoader(model);
  return { heavy: true, command, extra: extraForLoader(loader), model, loader: loader || null };
}

// What to tell the user before a heavy install: the download size and, on Windows, the CPU-wheel caveat.
function installNotice(plan, spec, dir, platform = process.platform) {
  const size = DOWNLOAD_SIZE[platform] || DOWNLOAD_SIZE.linux;
  const lines = [
    `\`${plan.command}\` loads a model, so it runs in its own environment with the extra "${plan.extra}": ${spec}`,
    `download: ${size}; the install output below is not hidden.`,
  ];
  if (platform === "win32") {
    lines.push(
      "Windows: pip installs the CPU-only build of PyTorch, so encoding runs on the CPU.",
      "For an NVIDIA GPU, install the CUDA build into this environment afterwards (the command for your CUDA " +
        `version is at https://pytorch.org/get-started/locally/ ; run it with ${venvBin(dir, "python")} -m pip). ` +
        "`optivision doctor` reports which build is installed.",
    );
  }
  return lines;
}

function pipArgs(spec, quiet) {
  return ["-m", "pip", "install", "--disable-pip-version-check", ...(quiet ? ["--quiet"] : []), spec];
}

// ------------------------------------------------------------------ environments

// Creates the environment for `spec` unless it exists; returns {dir, cli}. `notice(dir)` lists lines to print
// before a first install; `quiet` hides pip's output (the light environment does, the heavy one does not).
function ensureEnv(spec, { notice, quiet = true } = {}) {
  const dir = envDirFor(spec);
  const marker = path.join(dir, ".optivision-installed");
  const cli = venvBin(dir, "optivision");

  const ready = fs.existsSync(marker) && fs.existsSync(cli);
  if (ready && process.env.OPTIVISION_REINSTALL !== "1") return { dir, cli };

  const py = findPython();
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(path.dirname(dir), { recursive: true });
  if (notice) for (const line of notice(dir)) log(line);
  log(`first run: installing ${spec} with Python ${py.version} into ${dir}`);
  if (!runQuiet(py.exe, [...py.pre, "-m", "venv", dir])) {
    fs.rmSync(dir, { recursive: true, force: true });
    fail("could not create a virtual environment (on Debian/Ubuntu: apt install python3-venv)");
  }
  const venvPy = venvBin(dir, "python");
  if (!runQuiet(venvPy, pipArgs(spec, quiet))) {
    fs.rmSync(dir, { recursive: true, force: true });
    fail(`pip install ${spec} failed (see the output above)`);
  }
  fs.writeFileSync(marker, `${spec}\n`);
  log("installed; later runs start immediately");
  return { dir, cli };
}

const LOADER_SCRIPT = [
  "import json, sys",
  "info = None",
  "try:",
  "    from optivision.easy._resolve import resolve",
  "    info = resolve(sys.argv[1])",
  "except Exception:",
  "    try:",
  "        from optivision.adapters import MODELS",
  "        info = MODELS.get(sys.argv[1])",
  "    except Exception:",
  "        pass",
  "print(json.dumps({'loader': getattr(info, 'loader', None)}))",
].join("\n");

// The loader name of a model, asked of the light environment (which has the registry); null when unknown.
function loaderOf(model) {
  const { dir } = ensureEnv(requirement());
  const r = spawnSync(venvBin(dir, "python"), ["-c", LOADER_SCRIPT, model], { encoding: "utf8", windowsHide: true });
  if (r.status !== 0 || !r.stdout) return null;
  try {
    const lines = r.stdout.trim().split(/\r?\n/);
    return JSON.parse(lines[lines.length - 1]).loader || null;
  } catch (e) {
    return null;
  }
}

function ensureInstalled() {
  return ensureEnv(requirement()).cli;
}

function ensureHeavy(plan) {
  const spec = heavyRequirement(plan.extra);
  return ensureEnv(spec, { notice: (dir) => installNotice(plan, spec, dir), quiet: false }).cli;
}

function main() {
  const args = process.argv.slice(2);
  const plan = route(args, loaderOf);
  const cli = plan.heavy ? ensureHeavy(plan) : ensureInstalled();
  const child = spawn(cli, args, { stdio: "inherit" });
  for (const sig of ["SIGINT", "SIGTERM"]) {
    process.on(sig, () => child.kill(sig));
  }
  child.on("error", (e) => fail(`could not start ${cli}: ${e.message}`));
  child.on("exit", (code, signal) => {
    if (signal) process.kill(process.pid, signal);
    else process.exit(code === null ? 1 : code);
  });
}

if (require.main === module) main();

module.exports = {
  HEAVY_COMMANDS,
  commandOf,
  envDirFor,
  extraForLoader,
  heavyRequirement,
  indexInfo,
  installNotice,
  optionValue,
  pipArgs,
  positionals,
  requirement,
  route,
  withExtra,
};
