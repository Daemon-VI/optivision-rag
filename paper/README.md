# Paper — submission guide

`optivision.tex` is an IEEE conference paper built from the measured ablations in
[`../docs/RESULTS.md`](../docs/RESULTS.md).

`optivision.pdf` in this folder is the current build: 8 pages, compiled with
tectonic, no errors and no overfull boxes. Regenerate it with `tectonic -X compile
optivision.tex` from this directory after any edit.

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

**Compile in Normal mode, not Fast [draft].** Overleaf's draft mode renders every
`\includegraphics` as an empty box containing its own filename, which looks exactly like
missing figure files and is not. The setting is behind the arrow next to *Recompile*.
Fig. 1 still draws correctly in draft mode because it is built from `\fbox` rules rather
than an image — if Fig. 1 looks right and Figs. 2–4 are boxes, that is draft mode.

Figures are regenerated from the benchmark JSON, so they never drift from the numbers:

```bash
python paper/make_figs.py        # from the repo root
```

That writes six PDFs: `tradeoff`/`sweep` for E1, `tradeoff_colpali`/`sweep_colpali` for
E2, and `scale.pdf` which puts both experiments on one axes. Only three of them are
used by the current `.tex` (see *Files* below); the rest are kept because they are the
per-experiment views a reviewer may ask for during revision.

## Before you submit — checklist

- [ ] **Fix the e-mail addresses.** The ones in the author block are inferred from roll
      numbers and are *not verified*. There is a `TODO` comment above them.
- [ ] **Check the guide's name and title** are exactly as they should appear, and confirm
      author order with everyone listed. This is the one thing that cannot be fixed after
      acceptance.
- [x] ~~**Run the ViDoRe benchmark** and update the numbers.~~ Done: four splits,
      1,780 pages, 1,325 queries. See `../reports/colpali_*/` and
      `bash ../scripts/run_bench_gpu.sh` to reproduce.
- [x] ~~**Recompile the PDF.**~~ Done: 8 pages, clean compile, committed.
- [ ] **Check the venue's page limit.** The ColPali section adds two tables and a
      figure; the E1 sweep figure was dropped to compensate, since Table III now carries
      the sweep across all four splits. Recount after recompiling. To cut further,
      Section VI-A can lose its first paragraph without losing the argument.
- [ ] **Compile in Normal mode, not Fast [draft].** In draft mode every figure renders
      as a box containing its filename. Use the arrow beside Overleaf's Recompile button
      to check.
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

## The run that closed two limitations, and the one that would close the third

`scripts/run_bench_gpu.sh` reproduces the whole ablation with **ColPali-v1.3 on real
ViDoRe pages** on a free Kaggle/Colab T4. It has been run: four splits, 1,780 pages,
1,325 queries, archived under `../reports/colpali_*/`. Two of the paper's limitations —
"the corpus is generated, not scanned" and "ColSmol-256M is not ColPali-3B" — were
retired by it, and both would have been the first things a reviewer named.

It did not confirm the paper. It reversed it. The E1 claim was that the one-bit codec is
where retrieval quality is paid; under ColPali the codec costs at most 3.7% and what
fails instead is the pruning premise, because a ViDoRe page is not mostly blank paper.
That reversal is now the paper's contribution rather than a problem with it. Two things
about the run are worth knowing before you touch the numbers:

- **`vidore/infovqa_test_subsampled` is the adversarial split** — infographics, dense
  ink, nothing for a saliency detector to discard. It was included deliberately as the
  case that could break the pruning premise, and it did, though not in the way the
  premise fails loudly: spatial pruning returns **1.03x** there against 1.42x on the
  `energy` split, and costs almost no quality doing it. The detector is not wrong, it
  simply has nothing to remove. That is the finding, and it belongs in the paper rather
  than in a drawer.
- **The checkpoint matters.** `vidore/colpali-v1.3` is adapter-only; loading it without
  the base weights leaves `custom_text_proj` randomly initialised and yields a complete,
  plausible, meaningless table. The config pins `vidore/colpali-v1.3-merged`, and
  `src/optivision/encoders/colvlm.py` raises rather than proceeding if any parameter is
  still on the meta device. Do not relax that guard to make a run start.

**The next run, if you want one.** The remaining limitation is that encoder scale and
corpus provenance change together in our design, so the paper can say the attribution
reverses between the two but not what drives it. Running ColPali-v1.3 on the *generated*
corpus separates them, and it is one cell on the same T4:

```bash
optivision make-corpus data/corpus --docs 20        # seed 7, deterministic
optivision bench data/corpus/pdfs data/corpus/queries.json --config configs/colpali.yaml
```

`data/` is gitignored, so the corpus is regenerated rather than downloaded; the default
seed reproduces the exact pages E1 used. If the reversal follows the encoder, the claim
is about retrieval margin; if it follows the corpus, it is about page sparsity. Either
answer sharpens Section VI, and neither weakens the paper as written.

## Where the numbers come from

Every number in the paper traces to a `benchmark.json` the harness wrote directly —
nothing was transcribed by hand, and `make_figs.py` reads the same files the tables do.
The claims and their sources:

| claim in the paper | source |
|---|---|
| **E1** — Table I, Fig. 3, the E1 half of Fig. 4 | `reports/colsmol/benchmark.json` |
| **E2** — Table II, the E2 half of Fig. 4 | `reports/colpali_docvqa_test_subsampled/benchmark.json` |
| **E2** — Table III, the four-split retentions | all four `reports/colpali_*/benchmark.json` |
| 1,780 pages / 1,325 queries across the four splits | `corpus.n_pages` + `corpus.n_queries` in each of the four |
| query-encode and MaxSim latencies | `query_encode_ms` and `rows[].query_ms_p50` |
| 90.12 KB/page RAM before blocked scoring; 0.93x budget tracking | `docs/IMPROVEMENTS.md` §1 |
| int8 cosine error 3.28e-4 → 8.20e-5; mean abs component 0.071 | `docs/IMPROVEMENTS.md` §2 |
| saliency weights 0.6/0.4, threshold 0.02, dilate 1, min_keep 8 | `reports/colsmol/benchmark.json` → `config.pruning` |
| redundancy threshold 0.92, greedy clustering, centroid merge | `src/optivision/pruning/redundancy.py` |
| 233/768 patches, 233/875 vectors in Fig. 2 | `reports/figures/invoice_000_p1.png` |

E1 and E2 differ **only** in the encoder and the corpus — identical pruning config,
identical quantizers, identical exact-MaxSim scoring path, same code. That is the whole
basis for attributing the reversal to those two variables, so if you change a shared
default you invalidate both experiments at once and must rerun both.

The prose in `../docs/RESULTS.md` §*The same table on ColPali-3B and real ViDoRe pages*
is generated from the same JSON and is the longer version of Tables II and III.

If you change a config default, rerun the benchmark and `make_figs.py` before touching
the prose — otherwise the paper and the repo disagree, which is the single easiest thing
for a reviewer to catch and the hardest to explain.

## Files

```
optivision.tex             the paper
optivision.pdf             current build, 8 pages
make_figs.py               regenerates every figs/*.pdf from the benchmark JSON

used by the .tex:
figs/pruning_example.png   Fig. 2 — page / saliency / retained mask
figs/tradeoff.pdf          Fig. 3 — E1: compression ratio vs nDCG@5 retention
figs/scale.pdf             Fig. 4 — both experiments on one axes, arrows joining
                                    identical variants; the paper's key figure

generated, not currently referenced:
figs/sweep.pdf             E1 token-budget sweep — was Fig. 4 until Table III
                                    subsumed it across all four splits
figs/tradeoff_colpali.pdf  E2 alone, the counterpart to Fig. 3
figs/sweep_colpali.pdf     E2 token-budget sweep on docvqa — the 25.8-point
                                    drop from keep-50% to keep-10%, alone
```

Fig. 1 (the pipeline) is drawn in the `.tex` itself with a `tabular` of boxes, so there
is no external asset to keep in sync — which is also why it is the one figure that
still renders under draft mode.

If a reviewer asks to see E2's trade-off curve on its own, `figs/tradeoff_colpali.pdf`
and `figs/sweep_colpali.pdf` are already built and need only an `\includegraphics`.
