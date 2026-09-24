// OptiVision RAG VS Code extension.
//
// A front end over the `optivision` CLI (src/optivision/cli.py in the main repo). It never runs
// anything through a shell: every invocation is spawn(exe, argv, { shell: false }), so a query or
// path typed in the panel is never shell-interpreted.
//
// The CLI has no --json mode (it renders Rich tables for a human), so this extension does not try
// to parse structured state out of it the way a JSON-backed tool would. It streams raw stdout/
// stderr into the panel's log and, for `explain`, lists the PNG files it wrote afterwards.

import { ChildProcess, spawn } from "child_process";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import * as vscode from "vscode";

const IS_WIN = process.platform === "win32";

interface Cli {
  exe: string;
  fixed: string[];
  label: string;
}

let log: vscode.OutputChannel;
let statusBar: vscode.StatusBarItem;
let panel: vscode.WebviewPanel | undefined;
let cachedCli: Cli | undefined;
let cachedCliOk = false;

function config() {
  return vscode.workspace.getConfiguration("optivision");
}

// --------------------------------------------------------------------------- locating the CLI

function exeNames(base: string): string[] {
  return IS_WIN ? [`${base}.exe`, `${base}.cmd`, base] : [base];
}

function onPath(base: string): string | undefined {
  for (const dir of (process.env.PATH ?? "").split(path.delimiter)) {
    for (const name of exeNames(base)) {
      const full = path.join(dir, name);
      if (dir && fs.existsSync(full) && fs.statSync(full).isFile()) {
        return full;
      }
    }
  }
  return undefined;
}

/** `optivision` itself, then whichever Python launcher resolves on PATH. */
function candidates(): Cli[] {
  const configured = config().get<string[]>("command") ?? [];
  if (configured.length && configured[0]) {
    const [exe, ...fixed] = configured;
    const resolved = path.isAbsolute(exe) ? exe : onPath(exe) ?? exe;
    return [{ exe: resolved, fixed, label: configured.join(" ") }];
  }
  const out: Cli[] = [];
  const installed = onPath("optivision");
  if (installed) {
    out.push({ exe: installed, fixed: [], label: "optivision" });
  }
  // `python`/`python3` resolve through PATH, so they respect an active venv; `py -3` is the
  // Windows Launcher's own registry lookup and only goes last, as a fallback.
  const pythons = IS_WIN ? ["python", "python3", "py"] : ["python3", "python"];
  for (const cmd of pythons) {
    const exe = onPath(cmd);
    if (exe) {
      const fixed = cmd === "py" ? ["-3", "-m", "optivision.cli"] : ["-m", "optivision.cli"];
      out.push({ exe, fixed, label: `${cmd} -m optivision.cli` });
    }
  }
  return out;
}

/** The first candidate that answers `--help`. Cached until settings or folders change. */
async function resolveCli(): Promise<Cli | undefined> {
  if (cachedCli && cachedCliOk) {
    return cachedCli;
  }
  for (const cli of candidates()) {
    try {
      const { code, out } = await run(cli, ["--help"], workdir(), 15_000);
      if (code === 0 && /optivision/i.test(out)) {
        log.appendLine(`using ${cli.label}`);
        cachedCli = cli;
        cachedCliOk = true;
        return cli;
      }
    } catch (e) {
      log.appendLine(`not usable: ${cli.label}: ${(e as Error).message}`);
    }
  }
  cachedCliOk = false;
  return undefined;
}

// --------------------------------------------------------------------------- running the CLI

function workdir(): string {
  const cwd = config().get<string>("cwd")?.trim();
  if (cwd) {
    return cwd;
  }
  return vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? os.homedir();
}

/** Run to completion, capturing output. Used for the CLI probe, not for panel actions. */
function run(cli: Cli, args: string[], cwd: string, timeoutMs = 300_000): Promise<{ code: number; out: string }> {
  return new Promise((resolve, reject) => {
    const child = spawn(cli.exe, [...cli.fixed, ...args], { cwd, shell: false, windowsHide: true });
    let out = "";
    const timer = setTimeout(() => {
      killTree(child);
      reject(new Error(`timed out after ${timeoutMs / 1000}s`));
    }, timeoutMs);
    child.stdout?.on("data", (d: Buffer) => (out += d.toString("utf-8")));
    child.stderr?.on("data", (d: Buffer) => (out += d.toString("utf-8")));
    child.on("error", (e) => {
      clearTimeout(timer);
      reject(e);
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      resolve({ code: code ?? -1, out });
    });
  });
}

function killTree(child: ChildProcess | undefined): void {
  if (!child || child.exitCode !== null || child.pid === undefined) {
    return;
  }
  if (IS_WIN) {
    spawn("taskkill", ["/pid", String(child.pid), "/T", "/F"], { windowsHide: true });
  } else {
    child.kill("SIGINT");
  }
}

/** Run a panel action, streaming stdout/stderr into the webview's log as it arrives. */
function runStreaming(cli: Cli, args: string[], cwd: string, onChunk: (text: string) => void): Promise<number> {
  return new Promise((resolve) => {
    log.appendLine(`$ ${cli.label} ${args.join(" ")}`);
    onChunk(`$ ${[path.basename(cli.exe), ...cli.fixed, ...args].join(" ")}\n`);
    const child = spawn(cli.exe, [...cli.fixed, ...args], { cwd, shell: false, windowsHide: true });
    const pipe = (d: Buffer) => {
      const text = d.toString("utf-8");
      log.append(text);
      onChunk(text);
    };
    child.stdout?.on("data", pipe);
    child.stderr?.on("data", pipe);
    child.on("error", (e) => {
      const text = `could not start ${cli.exe}: ${e.message}\n`;
      log.append(text);
      onChunk(text);
      resolve(-1);
    });
    child.on("close", (code) => resolve(code ?? -1));
  });
}

// --------------------------------------------------------------------------- install

async function installCli(): Promise<void> {
  const pythons = IS_WIN ? ["python", "python3", "py"] : ["python3", "python"];
  const python = pythons.find((p) => onPath(p));
  if (!python) {
    const pick = await vscode.window.showWarningMessage(
      "No Python 3.10+ interpreter was found on PATH. Install Python, then retry.",
      "Open python.org",
    );
    if (pick) {
      void vscode.env.openExternal(vscode.Uri.parse("https://www.python.org/downloads/"));
    }
    return;
  }
  const args = python === "py" ? ["-3", "-m", "pip", "install", "optivision-rag"] : ["-m", "pip", "install", "optivision-rag"];
  const term = vscode.window.createTerminal({ name: "OptiVision RAG: install" });
  term.show();
  term.sendText(`${python} ${args.join(" ")}`);
  const again = await vscode.window.showInformationMessage(
    "Installing optivision-rag with pip in the terminal. Click Refresh once it finishes.",
    "Refresh",
  );
  if (again === "Refresh") {
    cachedCliOk = false;
    await refreshPanelStatus();
  }
}

// --------------------------------------------------------------------------- webview

type Action = "init-config" | "make-corpus" | "index" | "search" | "stats" | "explain" | "bench" | "fetch-vidore";

function buildArgs(action: Action, f: Record<string, string>): string[] {
  const cfg = f.config?.trim() || config().get<string>("configPath")?.trim() || "";
  const withConfig = (args: string[]) => (cfg ? [...args, "-c", cfg] : args);
  switch (action) {
    case "init-config":
      return ["init-config", f.out || "configs/my.yaml", "--backend", f.backend || "colsmol"];
    case "make-corpus":
      return [
        "make-corpus",
        f.out || "data/corpus",
        "--docs", f.docs || "20",
        "--pages", f.pages || "2",
        "--seed", f.seed || "7",
        "--code-scale", f.codeScale || "1.0",
      ];
    case "index": {
      const args = ["index", f.source, ...(f.append === "true" ? ["--append"] : [])];
      if (f.floatCache) args.push("--float-cache", f.floatCache);
      if (f.limit) args.push("--limit", f.limit);
      return withConfig(args);
    }
    case "search":
      return withConfig(["search", f.query, "--top-k", f.topK || "5"]);
    case "stats":
      return withConfig(["stats"]);
    case "explain": {
      const args = ["explain", f.source, "--out", f.out || "reports/figures", "--limit", f.limit || "4"];
      return withConfig(args);
    }
    case "bench": {
      const args = ["bench", f.source, f.queries, "--out", f.out || "reports"];
      if (f.limit) args.push("--limit", f.limit);
      if (f.sweep === "true") args.push("--sweep");
      if (f.codebook === "true") args.push("--codebook");
      if (f.cache) args.push("--cache", f.cache);
      args.push("--top-k", f.topK || "10");
      return withConfig(args);
    }
    case "fetch-vidore":
      return ["fetch-vidore", "--dataset", f.dataset || "vidore/syntheticDocQA_energy_test", "--out", f.out || "data/vidore", "--limit", f.limit || "100"];
  }
}

function resolveAgainst(cwd: string, p: string): string {
  return path.isAbsolute(p) ? p : path.join(cwd, p);
}

async function runAction(action: Action, fields: Record<string, string>): Promise<void> {
  if (!panel) {
    return;
  }
  const cli = await resolveCli();
  if (!cli) {
    panel.webview.postMessage({ type: "done", action, code: -1 });
    void offerInstall();
    return;
  }
  const cwd = workdir();
  const args = buildArgs(action, fields);
  const code = await runStreaming(cli, args, cwd, (text) => panel?.webview.postMessage({ type: "log", text }));

  let figures: string[] = [];
  if (action === "explain" && code === 0 && panel) {
    const dir = resolveAgainst(cwd, fields.out || "reports/figures");
    try {
      figures = fs
        .readdirSync(dir)
        .filter((f) => f.toLowerCase().endsWith(".png"))
        .map((f) => panel!.webview.asWebviewUri(vscode.Uri.file(path.join(dir, f))).toString());
    } catch (e) {
      log.appendLine(`could not read ${dir}: ${(e as Error).message}`);
    }
  }
  panel.webview.postMessage({ type: "done", action, code, figures });
}

async function offerInstall(): Promise<void> {
  const pick = await vscode.window.showWarningMessage(
    "optivision-rag is not installed (or not found). Install it now?",
    "Install", "Settings",
  );
  if (pick === "Install") {
    await installCli();
  } else if (pick === "Settings") {
    await vscode.commands.executeCommand("workbench.action.openSettings", "optivision.command");
  }
}

async function refreshPanelStatus(): Promise<void> {
  if (!panel) {
    return;
  }
  const cli = await resolveCli();
  panel.webview.postMessage({ type: "status", ok: !!cli, label: cli?.label });
  statusBar.text = cli ? `$(check) OptiVision RAG: ${cli.label}` : "$(warning) OptiVision RAG: not found";
  statusBar.tooltip = cli ? `Using ${cli.label}` : "Click to install optivision-rag";
  statusBar.command = cli ? "optivision.openPanel" : "optivision.installCli";
}

function openPanel(context: vscode.ExtensionContext): void {
  if (panel) {
    panel.reveal();
    return;
  }
  panel = vscode.window.createWebviewPanel("optivisionRag", "OptiVision RAG", vscode.ViewColumn.Active, {
    enableScripts: true,
    retainContextWhenHidden: true,
    localResourceRoots: [vscode.Uri.file(workdir()), vscode.Uri.file(os.homedir())],
  });
  panel.iconPath = vscode.Uri.joinPath(context.extensionUri, "media", "icon.png");
  panel.webview.html = renderHtml();
  panel.onDidDispose(() => (panel = undefined));
  panel.webview.onDidReceiveMessage(async (msg) => {
    if (msg.type === "run") {
      await runAction(msg.action, msg.fields ?? {});
    } else if (msg.type === "install") {
      await installCli();
    } else if (msg.type === "refresh") {
      await refreshPanelStatus();
    }
  });
  void refreshPanelStatus();
}

function renderHtml(): string {
  const field = (name: string, label: string, placeholder = "", value = "") =>
    `<label>${label}<input name="${name}" placeholder="${placeholder}" value="${value}"></label>`;
  const checkbox = (name: string, label: string) => `<label class="chk"><input type="checkbox" name="${name}">${label}</label>`;

  return /* html */ `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  :root { color-scheme: light dark; }
  body { font-family: var(--vscode-font-family); padding: 0 16px 24px; color: var(--vscode-foreground); }
  h2 { margin-top: 28px; border-bottom: 1px solid var(--vscode-panel-border); padding-bottom: 4px; }
  #status { padding: 8px 0; }
  #status.bad { color: var(--vscode-errorForeground); }
  fieldset { border: 1px solid var(--vscode-panel-border); border-radius: 4px; margin: 8px 0 16px; padding: 12px; }
  legend { opacity: 0.8; }
  label { display: block; margin: 6px 0; font-size: 12px; }
  label.chk { display: inline-block; margin-right: 16px; }
  label.chk input { margin-right: 4px; }
  input[type=text], input:not([type]) { width: 100%; box-sizing: border-box; padding: 4px 6px; margin-top: 2px;
    background: var(--vscode-input-background); color: var(--vscode-input-foreground);
    border: 1px solid var(--vscode-input-border, transparent); border-radius: 2px; }
  button { margin-top: 8px; padding: 4px 14px; cursor: pointer; background: var(--vscode-button-background);
    color: var(--vscode-button-foreground); border: none; border-radius: 2px; }
  button:hover { background: var(--vscode-button-hoverBackground); }
  details summary { cursor: pointer; font-weight: 600; margin: 20px 0 8px; }
  #log { white-space: pre-wrap; font-family: var(--vscode-editor-font-family); font-size: 12px;
    background: var(--vscode-textCodeBlock-background); border-radius: 4px; padding: 10px; max-height: 320px;
    overflow-y: auto; margin-top: 8px; }
  #figures { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; }
  #figures img { max-width: 260px; border: 1px solid var(--vscode-panel-border); border-radius: 4px; }
  .row { display: flex; gap: 10px; }
  .row > label { flex: 1; }
</style>
</head>
<body>
  <h1>OptiVision RAG</h1>
  <div id="status">checking for the CLI…</div>

  <h2>1. Init config</h2>
  <fieldset data-action="init-config">
    ${field("out", "Config path", "configs/my.yaml", "configs/my.yaml")}
    <label>Backend
      <select name="backend">
        <option value="colsmol">colsmol</option>
        <option value="colpali">colpali</option>
        <option value="colqwen2">colqwen2</option>
        <option value="synthetic">synthetic (offline, no downloads)</option>
      </select>
    </label>
    <button data-run>Write config</button>
  </fieldset>

  <h2>2. Make corpus</h2>
  <fieldset data-action="make-corpus">
    ${field("out", "Output directory", "data/corpus", "data/corpus")}
    <div class="row">${field("docs", "Docs", "20", "20")}${field("pages", "Pages/doc", "2", "2")}${field("seed", "Seed", "7", "7")}${field("codeScale", "Code scale", "1.0", "1.0")}</div>
    <button data-run>Generate</button>
  </fieldset>

  <h2>3. Index</h2>
  <fieldset data-action="index">
    ${field("source", "Source folder / file", "data/corpus/pdfs")}
    ${field("config", "Config (blank = setting default)", "configs/colsmol.yaml")}
    <div class="row">${field("floatCache", "Float cache (optional)")}${field("limit", "Limit pages (optional)")}</div>
    ${checkbox("append", "Append to existing index")}
    <button data-run>Build index</button>
  </fieldset>

  <h2>4. Search</h2>
  <fieldset data-action="search">
    ${field("query", "Query", "renewal of vehicle insurance policy")}
    <div class="row">${field("config", "Config (optional)")}${field("topK", "Top K", "5", "5")}</div>
    <button data-run>Search</button>
  </fieldset>

  <h2>5. Stats</h2>
  <fieldset data-action="stats">
    ${field("config", "Config (optional)")}
    <button data-run>Show stats</button>
  </fieldset>

  <h2>6. Explain</h2>
  <fieldset data-action="explain">
    ${field("source", "Source folder / file", "data/corpus/pdfs")}
    <div class="row">${field("config", "Config (optional)")}${field("out", "Figures out dir", "reports/figures", "reports/figures")}${field("limit", "Pages", "4", "4")}</div>
    <button data-run>Render figures</button>
    <div id="figures"></div>
  </fieldset>

  <details>
    <summary>Advanced</summary>

    <h2>Bench (ablation table — can take a while)</h2>
    <fieldset data-action="bench">
      ${field("source", "Corpus folder", "data/corpus/pdfs")}
      ${field("queries", "queries.json", "data/corpus/queries.json")}
      <div class="row">${field("config", "Config (optional)")}${field("out", "Out dir", "reports", "reports")}</div>
      <div class="row">${field("limit", "Limit pages (optional)")}${field("topK", "Top K", "10", "10")}${field("cache", "Encode cache .npz (optional)")}</div>
      ${checkbox("sweep", "Keep-ratio sweep")} ${checkbox("codebook", "Codebook rows")}
      <button data-run>Run benchmark</button>
    </fieldset>

    <h2>Fetch a real ViDoRe split</h2>
    <fieldset data-action="fetch-vidore">
      ${field("dataset", "HF dataset id", "vidore/syntheticDocQA_energy_test", "vidore/syntheticDocQA_energy_test")}
      <div class="row">${field("out", "Out dir", "data/vidore", "data/vidore")}${field("limit", "Limit", "100", "100")}</div>
      <button data-run>Download</button>
    </fieldset>
  </details>

  <h2>Log</h2>
  <div id="log"></div>

<script>
const vscode = acquireVsCodeApi();
const statusEl = document.getElementById('status');
const logEl = document.getElementById('log');
const figuresEl = document.getElementById('figures');

document.querySelectorAll('button[data-run]').forEach((btn) => {
  btn.addEventListener('click', () => {
    const fieldset = btn.closest('fieldset');
    const action = fieldset.dataset.action;
    const fields = {};
    fieldset.querySelectorAll('input, select').forEach((el) => {
      fields[el.name] = el.type === 'checkbox' ? String(el.checked) : el.value;
    });
    logEl.textContent = '';
    if (action === 'explain') { figuresEl.innerHTML = ''; }
    btn.disabled = true;
    vscode.postMessage({ type: 'run', action, fields });
  });
});

window.addEventListener('message', (event) => {
  const msg = event.data;
  if (msg.type === 'status') {
    statusEl.className = msg.ok ? '' : 'bad';
    statusEl.textContent = msg.ok ? ('using ' + msg.label) :
      'optivision-rag was not found. Run "OptiVision RAG: Install via pip" from the Command Palette, or set optivision.command.';
  } else if (msg.type === 'log') {
    logEl.textContent += msg.text;
    logEl.scrollTop = logEl.scrollHeight;
  } else if (msg.type === 'done') {
    document.querySelectorAll('button[data-run]').forEach((b) => (b.disabled = false));
    logEl.textContent += '\\n[exit code ' + msg.code + ']\\n';
    logEl.scrollTop = logEl.scrollHeight;
    if (msg.action === 'explain' && msg.figures && msg.figures.length) {
      figuresEl.innerHTML = msg.figures.map((u) => '<img src="' + u + '">').join('');
    }
  }
});

vscode.postMessage({ type: 'refresh' });
</script>
</body>
</html>`;
}

// --------------------------------------------------------------------------- activation

export function activate(context: vscode.ExtensionContext): void {
  log = vscode.window.createOutputChannel("OptiVision RAG");
  statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 50);
  statusBar.text = "$(sync~spin) OptiVision RAG";
  statusBar.command = "optivision.openPanel";
  statusBar.show();

  context.subscriptions.push(
    log,
    statusBar,
    vscode.commands.registerCommand("optivision.openPanel", () => openPanel(context)),
    vscode.commands.registerCommand("optivision.installCli", installCli),
    vscode.commands.registerCommand("optivision.showLog", () => log.show()),
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration("optivision")) {
        cachedCliOk = false;
        void refreshPanelStatus();
      }
    }),
    vscode.workspace.onDidChangeWorkspaceFolders(() => {
      cachedCliOk = false;
      void refreshPanelStatus();
    }),
  );

  void (async () => {
    const cli = await resolveCli();
    statusBar.text = cli ? `$(check) OptiVision RAG: ${cli.label}` : "$(warning) OptiVision RAG: not found";
    statusBar.tooltip = cli ? `Using ${cli.label}. Click to open the control panel.` : "Click to install optivision-rag";
    statusBar.command = cli ? "optivision.openPanel" : "optivision.installCli";
  })();
}

export function deactivate(): void {
  // spawned child processes end on their own when the CLI command finishes; nothing to tear down
}
