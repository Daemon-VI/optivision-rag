// Smoke test: install through the launcher into a throwaway home, then run the CLI.
// Point OPTIVISION_PIP_SPEC at a local wheel to test before the PyPI upload exists.
"use strict";
const { spawnSync } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const home = fs.mkdtempSync(path.join(os.tmpdir(), "optivision-npm-"));
const work = fs.mkdtempSync(path.join(os.tmpdir(), "optivision-work-"));
const env = { ...process.env, OPTIVISION_HOME: home };
const bin = path.join(__dirname, "..", "bin", "optivision.js");
const pkg = require("../package.json");

function cli(args) {
  const r = spawnSync(process.execPath, [bin, ...args], { env, cwd: work, encoding: "utf8" });
  if (r.status !== 0) {
    console.error(r.stdout, r.stderr);
    throw new Error(`optivision ${args.join(" ")} exited ${r.status}`);
  }
  return r.stdout;
}

try {
  const version = cli(["--version"]).trim();
  if (!version.includes(pkg.version)) {
    throw new Error(`version mismatch: npm ${pkg.version}, cli said ${version}`);
  }
  cli(["make-corpus", "data/corpus", "--docs", "3", "--pages", "1"]);
  cli(["index", "data/corpus/pdfs", "-c", "synthetic"]);
  const hits = cli(["search", "insurance policy", "-c", "synthetic"]);
  if (!hits.trim()) throw new Error("search printed nothing");
  console.log(hits);
  console.log(`smoke ok: ${version}`);
} finally {
  fs.rmSync(home, { recursive: true, force: true });
  fs.rmSync(work, { recursive: true, force: true });
}
