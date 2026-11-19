# Easy mode

Status: 2026-11-13, branch `easy-mode`. `optivision.easy` turns a folder of PDFs into a compressed,
searchable index with three calls: `options`, `build`, `search`. You give it a model and a target. It
picks the storage layout for you.

Every number in this guide comes from our own validation sets or from a run on one machine. None of
them has been checked on new documents. The package says so itself, and prints this sentence under
every plan and every build (it is read from `presets/easy_ledger.json`):

> These numbers come from our tests. They have not yet been checked on new documents; that check is registered (PREREG_EASY.md).

Code blocks tagged `py` need a model and, for a real run, a GPU. `tests/test_easy_notebook.py` runs
them on a CPU with a small stand-in encoder in place of the model. Blocks tagged `python` run as
written with no model. The outputs below were copied from those runs: the stand-in encoder changes
the lines that depend on the model, and each such place says so.

## Contents

1. [What it does](#1-what-it-does)
2. [From install to a first search](#2-from-install-to-a-first-search)
3. [Reading the build report](#3-reading-the-build-report)
4. [Reference: options and arguments](#4-reference-options-and-arguments)
5. [Check, add and upgrade](#5-check-add-and-upgrade)
6. [Command line](#6-command-line)
7. [Integrations that are on](#7-integrations-that-are-on)
8. [Limitations](#8-limitations)
9. [What the numbers mean](#9-what-the-numbers-mean)

## 1. What it does

You give it a model and a target. The target is the share of the model's full-precision search quality
(nDCG@5) to keep. The default is 0.97. It does not mean "97% of answers are right".

It then picks the most compressed storage layout whose measured quality clears the target. The choice
comes from a catalogue: a fixed ladder of layouts (8-bit, centred 4-bit and 1-bit codes, each with
optional merging of similar vectors) that we measured on public benchmarks, for each model we
measured. For each layout the catalogue holds an interval for the share of quality kept. The rule uses
the **lowest kept**: the low end of that interval on the model's worst dataset. A layout qualifies when
its lowest kept clears the target plus a margin. The margin is zero up to 500 pages, the size of the
collections we measured, and grows with the size of your folder above that. A chain rule adds a
second condition: a layout qualifies only if every lighter layout in its chain also does. Among the
layouts that qualify, the rule takes the one with the highest ratio. 1-bit layouts are open only to
folders that the practice check can test (see section 3).

Sizes and ratios are quoted against float16, the uncompressed size. 8-bit storage is about 2x on that
scale. (The README quotes ratios against float32, where the same 8-bit layout reads 3.9x.)

Three limits shape everything below:

- The catalogue numbers are measurements on our benchmarks. They are not a promise for your documents.
- Practice questions made from your PDFs can only warn. They do not change the layout.
- The only way to a smaller layout than the catalogue allows is your own real questions, through
  `upgrade()`.

## 2. From install to a first search

Install the package and the loader your model needs. `optivision doctor --model MODEL` prints the
line for a given model. For the example model:

```bash
pip install "optivision-rag[easy-st]"
```

Plan first. This needs no GPU and downloads nothing:

```python
import optivision.easy as ov

plan = ov.options("vidore/colqwen2-v1.0-hf", target=0.97, pages=20)
print(plan)
```

```text
vidore/colqwen2-v1.0-hf, target 97%, 20 pages (as given)
layout       merged (1 in 4 vectors) + 8-bit, layout="hm(0.25) > int8"; close call  [our benchmarks: ViDoRe V1 DocVQA 500 pages, InfoVQA 500 pages]
lowest kept  97.2% of full-precision search quality (nDCG@5)  [our benchmarks: ViDoRe V1 DocVQA 500 pages, InfoVQA 500 pages]
close call   clears 97% by under half a point. We use it by default.
             Safer: layout="int4(mean)" (centred 4-bit), 48.3-49.4 MB, 3.8x, lowest kept 98.0%  [our benchmarks: ViDoRe V1 DocVQA 500 pages, InfoVQA 500 pages]
size         about 487.1 KB of vectors, 7.8x smaller than float16  [estimate: our bytes per page x your pages]
encoding     about 42.6-87.5 s on one T4  [estimate: our Kaggle run, 0.61-0.65 s/page]
temporary    3.8 MB for the float16 copy in .ov_work  [estimate: our bytes per page x your pages]
free         5.7 GB reported by that drive  [counted]
if wrong     re-encoding costs the same time; keep_float16=True keeps the 3.8 MB copy so a change needs none  [estimate: our bytes per page x your pages]
status       These numbers come from our tests. They have not yet been checked on new documents; that check is registered (PREREG_EASY.md).
Next: ov.build("my_pdfs/", model="vidore/colqwen2-v1.0-hf", target=0.97)
```

(The temporary-folder path is shortened. The encode time is an estimate from our Kaggle runs on a T4.)

Read it from the top. The plan picks one layout for this model and target. "Lowest kept" is the
catalogue number for that layout. A **close call** means it clears the target by under half a point,
and the plan names a safer layout. Everything on a line carries one tag in brackets that says where
the number comes from (section 9).

Put your PDFs in a folder. To try the package without PDFs, copy the bundled 20-page sample (public
domain text, written for this package):

```py
import pathlib
import shutil

import optivision.easy as ov

pathlib.Path("my_pdfs").mkdir(exist_ok=True)
shutil.copy(pathlib.Path(ov.__file__).parent / "data" / "sample.pdf", "my_pdfs/sample.pdf")
```

Build the index. It encodes every page, picks the layout, saves one folder `my_pdfs.ovx` next to your
folder and prints a summary:

```py
index = ov.build("my_pdfs", model="vidore/colqwen2-v1.0-hf", target=0.97)
```

On a Colab T4, a real model prints the plan and a progress line, then the summary below. This summary
comes from a CPU run with the stand-in encoder on the sample PDF. The stand-in gives each page about
as many vectors as it has distinct words, far fewer than a real page model gives. That is why the
layout line differs from the plan above (see the Note line) and why `Search` is so short. Paths are
shortened:

```text
Model        vidore/colqwen2-v1.0-hf, sentence-transformers, float16 as in our run; device not recorded  [counted]
Wiring       12 pages, each found by its own sentence: 12 of 12. OK.  [counted]
Pages        20 from 1 files; 20 have a text layer; 0 files skipped, see report()  [counted]
Layout       target 97%: centred 4-bit. Lowest kept on our benchmarks 98.0%  [our benchmarks: vidore/docvqa_test_subsampled, vidore/infovqa_test_subsampled]
Note         your pages have far fewer vectors than the pages we measured merging on, so merged layouts are not used
Note         unchecked: no practice questions, so no 1-bit layouts
Unchecked    too few pages with a text layer for practice questions; 1-bit layouts are not used
Not checked  your real questions (queries=); your pages (no practice check)
Size         39.2 KB on disk in total: vectors 10.4 KB, page text, build record and the rest 28.9 KB  [counted]
Vectors      2.9x smaller than the float16 vectors (30.7 KB)  [counted]
Search       0.2 ms per query on this machine, timed on 20 sample queries  [counted]
Saved        .../my_pdfs.ovx    reload: ov.load(".../my_pdfs.ovx")
Status       These numbers come from our tests. They have not yet been checked on new documents; that check is registered (PREREG_EASY.md).
Next: index.search("buoy fish market turbidity harbour")
```

Search it. Each hit gives the page id, the file, the 1-based page number, a score and the page text.
`hit.image()` re-renders the page as a PIL image, ready for a vision-language model:

```py
hits = index.search("How do I get a golden yellow dye on wool?", k=3)
for hit in hits:
    print(hit.page, hit.id, round(hit.score, 2))
page = hits[0].image()
print(page.size)
```

```text
20 my_pdfs/sample.pdf::p20 6.5
11 my_pdfs/sample.pdf::p11 5.08
2 my_pdfs/sample.pdf::p2 5.0
(1241, 1754)
```

Page 20 of the sample is the page about natural dyes. With the stand-in encoder this only shows that
the wiring of the calls works. It says nothing about how well any real model searches. Scores are
MaxSim inner products. They order the pages of one index and mean nothing across indexes.

Save the full report. It lists every number with its tag and writes `.html`, `.md` or `.json`:

```py
report = index.report()
report.save("my_report.md")
print([title for title, block in report.sections])
```

```text
['Summary', 'Layouts at this target', 'Size', 'Build record', 'Ratios against float32', 'Geometry diagnostics']
```

Open the index again later, in a new session. The model loads only when a text question needs it:

```py
index = ov.load("my_pdfs.ovx")
```

## 3. Reading the build report

Each line of the summary has a label, a sentence and a tag.

| Line | What it tells you |
|---|---|
| `Model` | The model, its loader, the dtype and the device. "float16 as in our run" means the build used the dtype of our measurement run. |
| `Wiring` | Before encoding, twelve pages are each searched with a sentence of their own. If the model does not find them, the build stops and writes nothing; run `ov.doctor()`. On a folder whose pages look alike, the check repeats on the bundled sample and says so. |
| `Pages` | Pages, files, pages with a text layer, files skipped. Blank pages and repeated pages are counted in `report()`. |
| `Layout` | The target, the layout and its lowest kept (a catalogue number). |
| `Close call` | The layout clears the target by under half a point. It is kept by default; the line names a safer layout and what it costs. |
| `Note`, `Unchecked` | Why a layout was ruled out, such as too few vectors per page, or no practice questions. An unchecked folder gets no 1-bit layouts. |
| `Not checked` | What the build did not test: your real questions, and your pages if no practice check ran. |
| `Size`, `Vectors` | Bytes on disk and the ratio against float16. These are counted, not estimated. |
| `Search` | Time per query on this machine, from 20 sample queries. |
| `Status` | The status sentence from section 1. |
| `Next` | The call to make next. Every printed block ends with one. |

Practice questions are made from your PDFs' text, when the folder has 60 or more pages with a text
layer (the sample has 20, so it is "unchecked" above). They can also come from OCR, a local vision-language
model or an API model (`probes=`). They are described as **not reliably easier or harder than real
questions**. The practice check currently only warns: it does not change the layout and it never
confirms the target. Section 8 gives the reason.

## 4. Reference: options and arguments

`ov.options(model=None, target=None, *, pages=None)`

| Call | Result |
|---|---|
| `ov.options()` | Every registry model, with its status, licence, full-precision score, measured encode time and float16 size, then a Start here block. |
| `ov.options(model)` | The pick for that model and target, plus `larger` lines for collections of 1,000, 2,500 and 20,000 pages. |
| `ov.options(model, pages=...)` | Adds size, temporary disk, free space and encode time. `pages` is a number or a folder; a folder is counted without rendering. |
| `.table()` | Every layout of the model with its status at this target: `pick`, `qualifies`, `close call`, `below target` or `needs practice questions`. |
| `.save("chart.html")` | A self-contained chart (or `.svg`) of the layouts, with a target slider in the `.html`. |

Targets run from 0.90 to 1.0. Outside that range `options` raises `ValueError`. A target below 0.95 is
marked "less tested". Above 0.99 the plan says that our benchmarks cannot separate layouts this close to
100%. For the example model the plan gave these picks (each from `ov.options(...)` at that target):

| Target | Pick | Ratio against float16 | Lowest kept |
|---|---|---|---|
| 0.995 | float16 | 1.0x | 100.0% |
| 0.99 | 8-bit | 1.9x | 99.7% |
| 0.97 | merged (1 in 4 vectors) + 8-bit (a close call) | 7.8x | 97.2% |
| 0.95 | merged (1 in 4 vectors) + centred 4-bit | 15.4x | 96.3% |

`ov.build(source, model=None, target=None, *, path=None, ...)`

| Argument | Meaning |
|---|---|
| `source` | A folder, a PDF, an image, a list of strings (text models), a list of `(n, d)` vector arrays or a `MultiVectorCorpus`. |
| `model` | A registry id, a short name or an `EncoderAdapter`. An adapter that is not in the registry gets the rule for models without rows of their own: codecs only, using the worst result over measured models of the same kind. |
| `target` | 0.97 for a new index. `accuracy=` is an alias. An existing index keeps its stored target. |
| `path` | Where the index goes. Default: `NAME.ovx` beside the folder. An existing index is brought up to date: only new and changed files are encoded. |
| `layout` | A hand pick, such as `"int4(mean)"`. Every check still runs. |
| `keep_float16` | Keep the float16 copy. Needed by `upgrade()` and by `check(queries=...)`. |
| `keep_text` | Keep the page text for `Hit.text` (default on). |
| `work_dir` | Where the temporary float16 copy goes. On Google Drive a build resumes in a new session. |
| `rebuild`, `prune` | `rebuild=True` re-encodes at a new target or layout. `prune=True` removes files that disappeared. |
| `probes` | Practice questions: `"text"` (default), `"vlm"`, `"api"` (needs `allow_upload=True`) or `None`. |
| `queries`, `answers` | Your real questions (60 or more) and optional answers. They can only make the layout safer. |
| `sample` | Build on about this many pages first; the full build reuses their encoding. |
| `max_disk` | A size such as `"2 GB"`: the build uses the highest target, in steps of 0.01, that fits. |
| `trust_remote_code` | Needed by models that run code from their own repository. Without it the build stops and names the repository. |
| `device`, `dtype`, `progress` | Where and how the model runs; `progress=False` is quiet. |

A different target on an existing index raises `TargetMismatch` unless you pass `rebuild=True`. An
unknown model raises `UnknownModel` and names the three closest ids.

## 5. Check, add and upgrade

**`index.check()`** is a spot check. It compares the build record with the model now loaded, looks for
drift, asks whether the layout still qualifies at today's page count and reruns the practice check from
stored data. It changes nothing. It is not a retest of search quality on your documents.

```py
check = index.check()
print(check)
print(check.ok)
```

```text
Index      .../my_pdfs.ovx
Layout     centred 4-bit (int4(mean)), target 97%  [counted]
Record     sentence-transformers differs: None at build, not installed now
Record     torch differs: 2.11.0+cpu at build, 2.11.0 now
model settings (commit, dtype, attention, page size, batch) are compared when the model loads
Drift      drift not checked (this index saved no drift pages)
Pages      20 today, 20 when the layout was fitted  [counted]
Size rule  the layout still qualifies at this size (qualifies)
Practice   not rerun: no practice questions were made for this index (unchecked)
Result     needs attention
Next: ov.doctor('vidore/colqwen2-v1.0-hf')
False
```

The two `Record` lines come from the stand-in encoder and from the library versions on the machine that
ran the example. On your machine they name only what differs. `check(queries=your_questions)` also
compares the stored layout with full precision on your questions and needs `keep_float16=True`. With
`answers=` it scores search quality instead of agreement. `check.apply()` carries out a step-down it
advised, and prints its time first. It never runs on its own.

**`index.add(folder)`** adds new and changed files and leaves the rest alone. It is the same as running
`build` again. Files that disappeared are kept unless you pass `prune=True`.

```py
sync = index.add("my_pdfs")
```

```text
Sync
Sync       nothing new: 20 pages, unchanged  [counted]
Next: index.search(...) as before
```

`sync` is a `SyncReport` with the lists `added`, `changed`, `missing` and `skipped`, the page count and a
`message`.

**`index.upgrade(questions)`** is the only way to a smaller layout than the catalogue allows. It runs
the unchanged `certify()` on your real questions, with an error level that halves on each attempt so the
total stays at 0.05. It needs the float16 copy. Here the index was built without it:

```py
questions = ["How do I get a golden yellow dye on wool?"] * 3
print(index.upgrade(questions))
```

```text
Upgrade    needs the float16 copy, which this index does not keep
rebuilding with keep_float16=True re-encodes every page: about 42.6-87.5 s  [estimate: our Kaggle run, 0.61-0.65 s/page]
nothing was spent: no attempt is counted
Next: ov.build(..., keep_float16=True, rebuild=True), then index.upgrade(queries)
```

With the float16 copy kept and 150 questions, a run on the sample printed:

```text
Questions  150 new questions to test  [counted]
Needed     at least 175 questions if no top result changes; median 286 at mean agreement 99%  [estimate: the betting test's own arithmetic at this error level; median from a seeded simulation]
with fewer, even a layout that changes no top result cannot pass; nothing was spent
Next: collect more real questions (load(..., log_queries=True) keeps them), then index.upgrade()
```

The number of questions needed comes from the test itself. With too few, the call refuses and counts no
attempt. Without `answers=` the reference is agreement with the full-precision top page. Agreement is
not search quality, and the output says which one was tested. `upgrade(..., apply=True)` rebuilds the
index in the certified layout. `index.explain()` prints the expert-API code that is equivalent to the
index, including the `certify` call that `upgrade()` makes.

## 6. Command line

| Command | Purpose |
|---|---|
| `optivision options [MODEL] [--target T] [--pages DIR_OR_N] [--html FILE] [--json]` | The plan, or every model. |
| `optivision build SOURCE --model ID [--target T] [--path P] [--work-dir D] [--keep-float16] [--sample N] [--queries FILE] [--answers FILE] [--max-disk SIZE] [--trust-remote-code] [--json]` | Build, or bring an index up to date. |
| `optivision find INDEX "question" [-k 5] [--json]` | Search. The model loads on every call, so use `ov.load` for many questions. |
| `optivision check INDEX [--queries FILE] [--answers FILE] [--apply] [--json]` | Check. Exits 1 when the check does not pass. |
| `optivision doctor [--model ID] [--json]` | Check this machine. Exits 1 when something needs fixing. |
| `optivision quickstart [FOLDER] [--yes --model ID --target T]` | Guided setup. Where nobody can answer questions, such as a Colab `!` cell, it prints the plan and the `build` line instead. |

```bash
optivision options vidore/colqwen2-v1.0-hf --target 0.95 --pages 20
```

```text
vidore/colqwen2-v1.0-hf, target 95%, 20 pages (as given)
layout       merged (1 in 4 vectors) + centred 4-bit, layout="hm(0.25) > int4(mean)"  [our benchmarks: ViDoRe V1 DocVQA 500 pages, InfoVQA 500 pages]
lowest kept  96.3% of full-precision search quality (nDCG@5)  [our benchmarks: ViDoRe V1 DocVQA 500 pages, InfoVQA 500 pages]
size         about 247.3 KB of vectors, 15.4x smaller than float16  [estimate: our bytes per page x your pages]
encoding     about 42.6-87.5 s on one T4  [estimate: our Kaggle run, 0.61-0.65 s/page]
temporary    3.8 MB for the float16 copy in .ov_work  [estimate: our bytes per page x your pages]
free         5.7 GB reported by that drive  [counted]
if wrong     re-encoding costs the same time; keep_float16=True keeps the 3.8 MB copy so a change needs none  [estimate: our bytes per page x your pages]
status       These numbers come from our tests. They have not yet been checked on new documents; that check is registered (PREREG_EASY.md).
Next: ov.build("my_pdfs/", model="vidore/colqwen2-v1.0-hf", target=0.95)
```

`ov.doctor()` prints the same machine check from Python. It reports the operating system, Python, the
torch CUDA build (a CPU-only torch on a machine with a GPU is the usual cause of a slow build), the GPU,
free disk, library versions and known problems of your model. With a model it also prints the install
line. `ov.glossary()` lists the terms used in the printed text.

## 7. Integrations that are on

The switches live in `presets/easy_ledger.json`. This section lists what is on at the time of writing
and what each switch was tested on.

**Qdrant export: on.** `index.export_qdrant(...)`, or `optivision.easy.integrations.qdrant.export(index, url, collection)`,
writes one multivector point per page. Qdrant scores it with MaxSim and the dot product.

```py
from optivision.easy.integrations import qdrant

export = qdrant.export(index, ":memory:", "my_pdfs")
print(export)
```

```text
Qdrant     20 pages written to my_pdfs as float32 multivectors (MaxSim, dot product)  [counted]
Size       61.4 KB of vector numbers sent; the index's own stored vectors are 9.7 KB  [counted]
Exactness  float32 storage of the decoded vectors and float32 queries: the index's own MaxSim arithmetic
Next: qdrant.search(client, "my_pdfs", query_vectors)
```

What was tested: Qdrant's top 10 matched the index's own exact search in 100% of queries, on synthetic
corpora, in qdrant-client's local mode, with float32 storage. The float16 option is outside that
test. A collection in float32 is larger than the index: Qdrant cannot hold the 8-, 4- or 1-bit codes,
so the export stores the decoded numbers at 4 bytes each (see the `Size` line). It needs
`pip install "optivision-rag[qdrant]"`.

**byaldi import: on.** `optivision.easy.integrations.byaldi.import_index(folder)` builds an index from
the page vectors of a byaldi index folder, with no GPU and without encoding again. What was tested: a
byaldi 0.0.x index folder written by the test's own writer, which follows byaldi's source, came back bit
for bit, and a search over it ranked as MaxSim over byaldi's tensors. It has **not** been run against
the byaldi package itself. Reading the `.pt` files needs torch.

**LangChain and LlamaIndex retrievers.** `optivision.easy.integrations.langchain.OptiVisionRetriever`
and `optivision.easy.integrations.llamaindex.OptiVisionRetriever` wrap `Index.search`. Each result
carries the page text, the page id, the file, the page number, the score and the index's evidence tags
in its metadata. Install them with `pip install "optivision-rag[langchain]"` or `pip install "optivision-rag[llamaindex]"`.

**Search box.** `index.demo()` opens a local Gradio search box on `127.0.0.1` (`pip install "optivision-rag[easy-ui]"`).
It does not log searches.

**Switched off.** These stay off until their tests pass:

| Feature | State |
|---|---|
| Small-memory layout | Off. Its test (V9) was run and not passed: its working set was above the bound. |
| Practice questions raising the layout one step | Off. It passed on text-layer and OCR questions and failed on vision-model questions, and it has no effect while the practice check only warns. |
| Document-type presets | Off. See section 8. |

## 8. Limitations

Read this before you rely on a number.

**The numbers are in-sample.** The catalogue and the decisions below come from our own validation sets:
ViDoRe V1 DocVQA and InfoVQA (500 pages each) for page models, SciFact for text models, and our
validation corpora. They have not been checked on new documents. The check is registered in advance and
has not run.

**What the validation run did not pass.** The run is written up in
`reports/easy/results/SUMMARY.md`, and each departure from the plan is in `reports/easy/DEVIATIONS.md`.
Not every test passed:

| Test | What it asked | Result |
|---|---|---|
| V1t | Does the text-model rule keep the target on 90% of units at 0.97 and at 0.95? | Failed. It held on 100% of 8 units at 0.97 and on 88% at 0.95. |
| V3 | Does refitting on a sample, and growth, move scores by 0.005 or less on 95% of pairs? | Missed its 95% target: 82.4% of 102 pairs. |
| V4 | Do practice questions score above real queries (in at least 70% of units)? | Failed: 33% of units for text-layer questions, 35% for OCR, 45% for vision-model questions. |
| V5b | Does the practice check stay quiet when the layout is fine? | Failed for text-layer questions (it fired on 11% of pairs, the limit was 10%). |
| V5c | Does it catch misses? | Failed for OCR questions (fired on 43% of 30 pairs) and vision-model questions (0% of 7). |
| V6 | Does the wiring check pass on every loader that loads, and fail on planted faults? | Incomplete. 6 of 7 loaded loaders passed. The planted-fault half did not run. |

Because V4 failed, practice questions are described as not reliably easier or harder than real
questions. Because V5b and V5c failed, the practice check only warns. Why: in validation, practice
questions did not reliably score above real ones, and the check either fired too often or missed real
losses, depending on the kind of question. A warning costs nothing and is still useful as a signal.

**Some units were not scored.** The two held-out `ColVec1-9b` units were too large for the scoring
machine. `colSmol-500M` failed its own startup check on the pharmaceuticals and industrial corpora, so
it has no vectors there. Because of this, V1a to V1d, V2 and V3 read "not scored" in the summary, even
though every unit that exists was scored on its own numbers.

**The 1-bit extra margin was removed.** The arm of the test that had no extra margin for 1-bit layouts
made no misses on 176 units. That is still an in-sample result.

**Document-type presets are not shipped.** The test asked whether the standard rule keeps low >= target
on each document type. It did for academic documents (32 of 32 units, 3 datasets, 4 models) and did not
for slides, reports or tables. It produced no type-specific numbers. The academic result is evidence,
not a rule: the package has no `type=` argument and the standard rule applies to every folder.

**Not tested:** other kinds of documents than those above; folders larger than the ones we measured
(the rule uses codecs only above 20,000 pages); a Qdrant server; the byaldi package; and any model the
catalogue has no row for. For those models the rule falls back to the worst result over measured models
of the same kind, and the output says so.

## 9. What the numbers mean

Every printed number has one tag. A line never mixes tags.

| Tag | Meaning | May change the layout |
|---|---|---|
| `[our benchmarks: ...]` | A fixed layout we measured on the named public benchmarks. Your documents may differ. | It sets the ceiling. |
| `[questions from your PDFs]` | Practice questions made from your PDFs' text. | It warns; it does not change the layout. |
| `[your questions: ...]` | Your real questions, scored against full precision (agreement) or against your answers (search quality). The tag says which. | Safer layouts; smaller ones only through `upgrade()`. |
| `[counted]` | Bytes, pages and times on your machine. | Never. |
| `[estimate: ...]` | Arithmetic from our bytes or our timings. | Never. |

How to read the main numbers:

- **Lowest kept** is a catalogue number, not a promise for your documents. It is the low end of the
  range we measured on the model's worst dataset.
- **Target 0.97** keeps at least 97% of the model's own nDCG@5. It does not mean 97% of answers are
  right.
- **Ratios** are against float16. Float16 is 1.0x, 8-bit is about 1.9x to 2.0x, and the README's
  float32 ratios are twice as large.
- **Close call** means the layout clears its needed line by under half a point.
- **`check()` is a spot check**, not a test of search quality on your documents.
- **`upgrade()` runs the unchanged `certify()`** on your questions. The statement that rule makes, and
  the assumptions it needs, are in [SELECTION_AND_CERTIFICATION.md](SELECTION_AND_CERTIFICATION.md).

The models behind the catalogue, and what was measured for each, are in [MODELS.md](MODELS.md). The
ideas (what a layout is, what retention means) are in [CONCEPTS.md](CONCEPTS.md). The expert API that
easy mode wraps is in [QUICKSTART.md](QUICKSTART.md).
