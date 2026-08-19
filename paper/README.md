# Paper — submission guide

`optivision.tex` is an IEEE conference paper built from the measured ablations in
[`../docs/RESULTS.md`](../docs/RESULTS.md).

> **`optivision.pdf` in this folder is stale.** It is the build from before the
> ColPali/ViDoRe results landed, and it still argues the superseded single-experiment
> claim. Recompile before you send it anywhere — see *Compiling* below.

## What the paper argues

Not "we built a compression pipeline and it compresses well" — that claim is true but
unremarkable, and a reviewer has seen it before. The paper argues something narrower and
more defensible:

> Systems that prune visual tokens *and* quantize them report the product of the two
> savings. Nobody reports the ratio of the two costs. We measured it twice — once with a
> 256M encoder on generated pages, once with ColPali-3B on ViDoRe — and got opposite
> answers from the same code. Under the small encoder the one-bit codec costs everything
> and pruning is free. Under the reference encoder the codec is nearly free, and what
> fails is the premise that a document page is mostly blank paper.

The reversal *is* the contribution, and it is a stronger one than the original
single-experiment claim. That version said "the codec is the binding constraint", which
the reference encoder shows to be false; this version says the attribution itself is a
property of the encoder and corpus, which is why publishing a compression ratio without
naming both is not enough. Do not soften either half during revision, and do not quietly
drop the E1 numbers — the paper only works because both experiments are reported.

Two details a reviewer will look for, both already in the text: Kendall $\tau$ falls just
as far under one-bit codes in *both* experiments (0.53 vs 0.59), which is what pins the
difference on the encoder's margin rather than on the codec; and the `energy` split's
above-100% retentions are called out as noise at 100 queries rather than presented as
compression improving retrieval.

## Compiling

**Overleaf (easiest, nothing to install).** New Project → Upload Project → zip this
folder. Set the compiler to pdfLaTeX in Menu → Settings. IEEEtran ships with Overleaf.

**Locally.** Needs a TeX distribution. If you do not have one, `tectonic` is a single
binary that downloads what it needs on first run:

```bash
# from paper/
tectonic -X compile optivision.tex
```

Figures are regenerated from the benchmark JSON, so they never drift from the numbers:

```bash
python paper/make_figs.py        # from the repo root
```

## Before you submit — checklist

- [ ] **Fix the e-mail addresses.** The ones in the author block are inferred from roll
      numbers and are *not verified*. There is a `TODO` comment above them.
- [ ] **Check the guide's name and title** are exactly as they should appear, and confirm
      author order with everyone listed. This is the one thing that cannot be fixed after
      acceptance.
- [x] ~~**Run the ViDoRe benchmark** and update the numbers.~~ Done: four splits,
      1,780 pages, 1,325 queries. See `../reports/colpali_*/` and
      `bash ../scripts/run_bench_gpu.sh` to reproduce.
- [ ] **Recompile the PDF.** The committed one predates the ColPali results.
- [ ] **Check the venue's page limit.** The ColPali section adds two tables and a
      figure, so recount pages after recompiling. To cut: the E1 sweep figure is the most
      expendable now that Table III carries the sweep across all four splits, and
      Section VI-A can lose its first paragraph without losing the argument.
- [ ] **Use the venue's own template** if it supplies one. Most IEEE conferences use
      stock `IEEEtran` and this file will drop straight in; some add a copyright notice
      block or a specific `\documentclass` option.
- [ ] **Check whether the venue is double-blind.** Most Indian IEEE/Springer conferences
      are not, but if yours is, strip the author block and the acknowledgment, and check
      that the GitHub URL does not appear anywhere.
- [ ] **Run a plagiarism check** (most institutes require a Turnitin report at
      submission). The text is original, but Related Work paraphrases cited papers
      closely by nature and self-overlap with your Stage-1 report will be flagged — that
      overlap is expected and explainable, but know the number before someone asks.

## The one run that most improves this paper

`notebooks/vidore_colpali_bench.ipynb` reproduces the whole ablation with **ColPali-v1.3
on real ViDoRe pages** using a free Colab/Kaggle T4. It finishes inside one free session.

This matters more than any amount of rewriting. Two of the paper's four limitations —
"the corpus is generated, not scanned" and "ColSmol-256M is not ColPali-3B" — exist only
because that run has not happened, and a reviewer will name both. The notebook's closing
cell lists exactly which parts of the `.tex` to change afterwards.

A specific thing to watch: `vidore/infovqa_test_subsampled` is infographics, which are
*not* mostly blank paper. It is the adversarial case for the pruning premise. If spatial
pruning survives it, the claim is much stronger than it currently is; if it collapses,
that is a genuine finding and belongs in the paper rather than in a drawer.

## Where the numbers come from

Every figure in the paper traces to `reports/colsmol/benchmark.json`, which the benchmark
harness writes directly — nothing was transcribed by hand. The claims and their sources:

| claim in the paper | source |
|---|---|
| the ablation table, all figures | `reports/colsmol/benchmark.json` |
| 90.12 KB/page RAM before blocked scoring; 0.93x budget tracking | `docs/IMPROVEMENTS.md` §1 |
| int8 cosine error 3.28e-4 → 8.20e-5; mean abs component 0.071 | `docs/IMPROVEMENTS.md` §2 |
| saliency weights 0.6/0.4, threshold 0.02, dilate 1, min_keep 8 | `reports/colsmol/benchmark.json` → `config.pruning` |
| redundancy threshold 0.92, greedy clustering, centroid merge | `src/optivision/pruning/redundancy.py` |
| 233/768 patches, 233/875 vectors in Fig. 2 | `reports/figures/invoice_000_p1.png` |

If you change a config default, rerun the benchmark and `make_figs.py` before touching
the prose — otherwise the paper and the repo disagree, which is the single easiest thing
for a reviewer to catch and the hardest to explain.

## Files

```
optivision.tex      the paper
optivision.pdf      current build
make_figs.py        regenerates figs/tradeoff.pdf and figs/sweep.pdf from the JSON
figs/tradeoff.pdf   Fig. 3 — compression ratio vs nDCG@5 retention, by quantizer
figs/sweep.pdf      Fig. 4 — the token-budget sweep
figs/pruning_example.png   Fig. 2 — page / saliency / retained mask
```

Fig. 1 (the pipeline) is drawn in the `.tex` itself with a `tabular` of boxes, so there
is no external asset to keep in sync.
