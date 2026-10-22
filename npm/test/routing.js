// Routing test: which environment each command runs in. No network, no Python, no install.
// `options` and `doctor` (and every older command) stay in the light environment; `build`, `find`, `check`
// and `quickstart` go to a heavy environment keyed by the extra, chosen from the model's loader.
"use strict";
const assert = require("assert");
const fs = require("fs");
const os = require("os");
const path = require("path");

const launcher = require("../bin/optivision.js");
const pkg = require("../package.json");

// The environment variables the launcher reads; each test starts from a clean slate.
const KEYS = ["OPTIVISION_EXTRAS", "OPTIVISION_PIP_SPEC", "OPTIVISION_HOME"];
const saved = Object.fromEntries(KEYS.map((k) => [k, process.env[k]]));
function clean(set = {}) {
  for (const k of KEYS) delete process.env[k];
  Object.assign(process.env, set);
}

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "optivision-routing-"));
const lookups = [];
// A stand-in for the registry lookup the real launcher asks of the light environment.
function lookup(model) {
  lookups.push(model);
  const table = {
    "vidore/colqwen2-v1.0-hf": "sentence-transformers",
    "vidore/colqwen2.5-v0.2": "colpali-engine:colqwen2.5",
    "TomoroAI/tomoro-colqwen3-embed-4b": "remote-code:tomoro",
  };
  return table[model] || null;
}

function writeIndex(name, files) {
  const dir = path.join(tmp, name);
  fs.mkdirSync(dir, { recursive: true });
  for (const [file, doc] of Object.entries(files)) {
    fs.mkdirSync(path.dirname(path.join(dir, file)), { recursive: true });
    fs.writeFileSync(path.join(dir, file), JSON.stringify(doc));
  }
  return dir;
}

try {
  // ---- the command and its arguments
  assert.strictEqual(launcher.commandOf(["build", "my_pdfs/", "--model", "x"]), "build");
  assert.strictEqual(launcher.commandOf(["--version"]), undefined);
  assert.strictEqual(launcher.optionValue(["build", "--model", "a/b"], "--model"), "a/b");
  assert.strictEqual(launcher.optionValue(["build", "--model=a/b"], "--model"), "a/b");
  assert.strictEqual(launcher.optionValue(["build", "--target", "0.9"], "--model"), undefined);
  assert.deepStrictEqual(launcher.positionals(["find", "-k", "3", "docs.ovx", "what?", "--json"]), ["docs.ovx", "what?"]);
  assert.deepStrictEqual(launcher.positionals(["build", "--model", "m", "folder", "--keep-float16"]), ["folder"]);

  // ---- the extra follows the loader
  assert.strictEqual(launcher.extraForLoader("sentence-transformers"), "easy-st");
  assert.strictEqual(launcher.extraForLoader("colpali-engine:colqwen2.5"), "easy-colpali");
  assert.strictEqual(launcher.extraForLoader("colpali-engine:colsmol"), "easy-colpali");
  assert.strictEqual(launcher.extraForLoader("remote-code:tomoro"), "easy-st");
  assert.strictEqual(launcher.extraForLoader("nemotron"), "easy-st");
  assert.strictEqual(launcher.extraForLoader(null), "easy-st");

  // ---- light commands: the older ones, options, doctor, help
  for (const args of [
    ["options"], ["options", "vidore/colqwen2-v1.0-hf", "--target", "0.97"], ["doctor"], ["doctor", "--model", "x"],
    ["--version"], ["--help"], ["models"], ["certify", "docs.npz"], ["index", "data/corpus/pdfs"], ["search", "q"],
    ["make-corpus", "data/corpus"], ["build", "--help"], ["find", "--help"], ["check", "--help"],
    ["quickstart", "--help"], [],
  ]) {
    assert.strictEqual(launcher.route(args, lookup).heavy, false, `light: ${args.join(" ")}`);
  }
  assert.strictEqual(lookups.length, 0, "a light command never asks the registry");

  // ---- build and quickstart: the model on the command line decides
  let plan = launcher.route(["build", "my_pdfs/", "--model", "vidore/colqwen2-v1.0-hf"], lookup);
  assert.deepStrictEqual([plan.heavy, plan.extra, plan.loader], [true, "easy-st", "sentence-transformers"]);
  plan = launcher.route(["build", "my_pdfs/", "--model=vidore/colqwen2.5-v0.2", "--target", "0.95"], lookup);
  assert.deepStrictEqual([plan.heavy, plan.extra], [true, "easy-colpali"]);
  plan = launcher.route(["quickstart", "my_pdfs/", "--model", "vidore/colqwen2.5-v0.2", "--yes"], lookup);
  assert.deepStrictEqual([plan.command, plan.extra], ["quickstart", "easy-colpali"]);
  plan = launcher.route(["build", "my_pdfs/", "--model", "TomoroAI/tomoro-colqwen3-embed-4b"], lookup);
  assert.strictEqual(plan.extra, "easy-st");
  plan = launcher.route(["quickstart"], lookup); // the wizard picks the model itself: Start here is sentence-transformers
  assert.deepStrictEqual([plan.heavy, plan.extra, plan.model], [true, "easy-st", undefined]);
  plan = launcher.route(["build", "my_pdfs/", "--model", "unknown/model"], lookup);
  assert.deepStrictEqual([plan.heavy, plan.extra, plan.loader], [true, "easy-st", null]);

  // ---- find and check: the index folder says which model it was built with
  const colpali = writeIndex("a.ovx", { "manifest.json": { info: { model: "vidore/colqwen2.5-v0.2", loader: "colpali-engine:colqwen2.5" } } });
  lookups.length = 0;
  plan = launcher.route(["find", colpali, "what drove costs?", "-k", "3"], lookup);
  assert.deepStrictEqual([plan.heavy, plan.command, plan.extra], [true, "find", "easy-colpali"]);
  assert.strictEqual(lookups.length, 0, "the loader in the manifest is used without a lookup");
  const stOnly = writeIndex("b.ovx", { "manifest.json": { record: { model: "vidore/colqwen2.5-v0.2" } } });
  plan = launcher.route(["check", stOnly], lookup);
  assert.deepStrictEqual([plan.extra, lookups], ["easy-colpali", ["vidore/colqwen2.5-v0.2"]]);
  const record = writeIndex("c.ovx", { "build.json": { loader: "sentence-transformers", model: "vidore/colqwen2-v1.0-hf" } });
  assert.strictEqual(launcher.route(["find", record, "q"], lookup).extra, "easy-st");
  // `my_pdfs` finds `my_pdfs.ovx`
  assert.strictEqual(launcher.route(["find", path.join(tmp, "a"), "q"], lookup).extra, "easy-colpali");
  // no index yet, or no readable manifest: the default
  assert.strictEqual(launcher.route(["find", path.join(tmp, "missing.ovx"), "q"], lookup).extra, "easy-st");
  fs.mkdirSync(path.join(tmp, "empty.ovx"));
  fs.writeFileSync(path.join(tmp, "empty.ovx", "manifest.json"), "{not json");
  assert.strictEqual(launcher.route(["check", path.join(tmp, "empty.ovx")], lookup).extra, "easy-st");

  // ---- the light requirement is unchanged
  clean();
  assert.strictEqual(launcher.requirement(), `optivision-rag[corpus]==${pkg.version}`);
  clean({ OPTIVISION_EXTRAS: "vlm,corpus" });
  assert.strictEqual(launcher.requirement(), `optivision-rag[vlm,corpus]==${pkg.version}`);
  clean({ OPTIVISION_EXTRAS: "" });
  assert.strictEqual(launcher.requirement(), `optivision-rag==${pkg.version}`);

  // ---- the heavy requirement adds the extra, and only that by default
  clean();
  assert.strictEqual(launcher.heavyRequirement("easy-st"), `optivision-rag[easy-st]==${pkg.version}`);
  assert.strictEqual(launcher.heavyRequirement("easy-colpali"), `optivision-rag[easy-colpali]==${pkg.version}`);
  clean({ OPTIVISION_EXTRAS: "vlm,corpus" });
  assert.strictEqual(launcher.heavyRequirement("easy-st"), `optivision-rag[vlm,corpus,easy-st]==${pkg.version}`);
  clean({ OPTIVISION_EXTRAS: "" });
  assert.strictEqual(launcher.heavyRequirement("easy-st"), `optivision-rag[easy-st]==${pkg.version}`);
  clean({ OPTIVISION_PIP_SPEC: "optivision-rag[corpus] @ file:///dist/optivision_rag-0.6.0-py3-none-any.whl" });
  assert.strictEqual(
    launcher.heavyRequirement("easy-st"),
    "optivision-rag[corpus,easy-st] @ file:///dist/optivision_rag-0.6.0-py3-none-any.whl",
  );
  assert.strictEqual(launcher.requirement(), process.env.OPTIVISION_PIP_SPEC, "the light spec is used as given");
  assert.strictEqual(launcher.withExtra("dist/x.whl", "easy-st"), "dist/x.whl[easy-st]");
  assert.strictEqual(launcher.withExtra("optivision-rag==0.6.0", "easy-st"), "optivision-rag[easy-st]==0.6.0");
  assert.strictEqual(launcher.withExtra("optivision-rag @ file:///x.whl", "easy-st"), "optivision-rag[easy-st] @ file:///x.whl");
  assert.strictEqual(launcher.withExtra("x[easy-st]==1", "easy-st"), "x[easy-st]==1");

  // ---- one environment per extra, none shared with the light one
  clean({ OPTIVISION_HOME: tmp });
  const dirs = [
    launcher.envDirFor(launcher.requirement()),
    launcher.envDirFor(launcher.heavyRequirement("easy-st")),
    launcher.envDirFor(launcher.heavyRequirement("easy-colpali")),
  ];
  assert.strictEqual(new Set(dirs).size, 3, dirs.join("\n"));
  assert.strictEqual(dirs[1], launcher.envDirFor(launcher.heavyRequirement("easy-st")), "keyed by the extra: stable");

  // ---- before a heavy install: the download size, and on Windows the CPU-wheel caveat
  clean({ OPTIVISION_HOME: tmp });
  const spec = launcher.heavyRequirement("easy-st");
  const heavy = { command: "build", extra: "easy-st" };
  for (const platform of ["linux", "darwin", "win32"]) {
    const text = launcher.installNotice(heavy, spec, path.join(tmp, "env"), platform).join("\n");
    assert.ok(/download: roughly [0-9. ]+ to [0-9. ]+ GB/.test(text), `${platform}: ${text}`);
    assert.ok(text.includes("not hidden"), "says the install output is shown");
    assert.ok(text.includes("easy-st") && text.includes(spec));
    assert.strictEqual(/CPU-only build of PyTorch/.test(text), platform === "win32", platform);
  }
  const win = launcher.installNotice(heavy, spec, path.join(tmp, "env"), "win32").join("\n");
  assert.ok(win.includes("https://pytorch.org/get-started/locally/") && win.includes("optivision doctor"));
  assert.ok(launcher.installNotice(heavy, spec, "d", "linux").join("\n").includes("3 to 4 GB"));

  // ---- the heavy install shows pip's output; the light one keeps --quiet
  assert.ok(!launcher.pipArgs(spec, false).includes("--quiet"));
  assert.ok(launcher.pipArgs(spec, true).includes("--quiet"));
  assert.strictEqual(launcher.pipArgs(spec, false).at(-1), spec);

  assert.deepStrictEqual([...launcher.HEAVY_COMMANDS].sort(), ["build", "check", "find", "quickstart"]);
  console.log("routing ok");
} finally {
  for (const k of KEYS) {
    if (saved[k] === undefined) delete process.env[k];
    else process.env[k] = saved[k];
  }
  fs.rmSync(tmp, { recursive: true, force: true });
}
