# Protenix-v2 inference speed strategy

## Objective and evidence

Increase throughput on the pipeline's A30/A100 workers while preserving sampling
budgets, structural-template support, VHH MSAs, confidence outputs and selection
rules. Start with overhead reductions, then evaluate Anthropic's Exact and Fast
implementations as serious candidates for production.

Anthropic's [technical report, section S1.7](https://www-cdn.anthropic.com/c03643714397d9d396fa1ce1794f5f9f7863a82c.pdf)
reports a 4.08x geometric-mean forward speedup for Protenix-v2 Fast; whole-pass
speedups are 2.39x for Exact and 3.19x for Fast. These are H100 measurements at
200-1,400 tokens, with 11 trunk cycles and five diffusion samples. On 251 interfaces
from 152 targets, default success is 66.1%, versus 64.5% for each optimized mode;
no paired accuracy change has a 95% interval excluding zero. This supports testing
Fast, but neither proves zero loss nor promises the same speed on our hardware.
The report also observes repeat-run differences in unmodified Protenix-v2 and
checks Exact equivalence separately under deterministic settings.

Sources: [article](https://www.anthropic.com/research/claude-uplifts-biomolecular-modeling),
[kit changes](https://github.com/anthropics/uplifting-biomolecular-modeling/blob/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/protenix_v2/CHANGES.md).
The initial audit compared kit commit `f4f62fa6592ae4938d49b1757bea0cfeff9f468e`,
fork commit `c5f0d8eb89923aebf03f4d56de2eab2ac3313e61`, and pipeline commit
`1d2314dcd5e61697a1172bd7abb981502a5e6e72`.

## Stage 1: minimal overhead changes (this patch)

1. Release the prediction dictionary after the synchronous dumper finishes,
   before the existing per-item allocator flush. Release the loop's reference on
   output failure too. This avoids carrying completed output tensors through the
   next forward pass, including subsequent seeds of the same candidate.
2. Keep cached allocations at confidence-head entry for inputs up to 2,000
   tokens. Preserve the original entry flush above that threshold and all
   large-input per-sample CPU-offload releases. Leave between-item and error-path
   flushes intact.

Only tensor lifetime and allocator policy change. No model operation, RNG call,
weight, precision setting, template, MSA, output field or sampling parameter changes.
Prediction release primarily improves memory headroom; allocator retention may
reduce runtime. Neither has a measured speedup yet. Reserved-memory and
fragmentation behaviour still need checking on the target GPUs.

Validation here: CPU execution of the production inference-loop source with
lightweight dependencies checks output handoff, prediction lifetime across two
items and two seeds, and continuation after a dump failure. The actual confidence
entry cleanup block is exercised at 400, 2,000 and 2,001 tokens and in training.
These checks establish control-flow behaviour, not numerical/GPU equivalence.

## Stage 2: remove startup and repeated preparation costs

| Work | Implementation boundary | Verification |
|---|---|---|
| Reuse compiled kernels | Pipeline SLURM template/runtime; preserve explicit cache overrides | Cold/warm timing, concurrent jobs, read-only/full-cache cases |
| Skip overwritten random initialization | Fork runner model construction, only with strict checkpoint loading | All loaded parameters and buffers match; seed reset occurs before stochastic featurization/inference |
| Reuse identical template calculations | Template embedder, scoped to one call and current pair representation | Compare complete feature slices; preserve addition order and all chain mappings |
| Reuse MSA pair weights across chunks | MSA block, only while its pair input remains unchanged | First establish that actual VHH MSA depth spans chunks; test output equality |

For compilation caches, use an architecture/software/source-keyed, per-user cache.
Avoid many array workers compiling into one fresh shared directory. Prefer a
prewarmed immutable cache copied to task-local writable storage, or verified
concurrency-safe cache sharing. Include Torch/CUDA/Triton/cuEquivariance versions,
GPU architecture and relevant kernel source version in its identity. Do not
enable kernel autotuning or change kernel selection as part of cache reuse.
Time cache copying too: it must save more than it costs.

Keep initialization zero/one constants and nonpersistent buffers intact. Reject
the optimization for partial/non-strict checkpoint loads. Model reuse across
candidates is a later option only if profiling shows construction is a material
cost; it would require deliberate config/cache reset and changes to job ownership.

## Stage 3: integrate Exact execution on the fork

Use a separate pinned experimental image and opt-in execution mode. Activate the
adapter before model/Pairformer imports in the pipeline's direct Python runner;
changing its historical CLI override alone does not cover the active VHH-MSA path.
Keep an explicit original-fork mode and log which optimizations actually execute.

Priorities are diffusion-step invariant reuse, exact elementwise fusion, sampler
CUDA graphs, then applicable Pairformer and triangle kernels. Profile first and
ablate one change at a time; a claimed active optimization with zero served calls
does not count as acceleration. Validate graph inputs, RNG order, cache invalidation
between seeds/candidates, and memory admission. Unsupported shapes should use the
original computation with an explicit log, never silently switch to Fast.

Compatibility tasks before enabling the kit as a whole:

- Preserve the fork's direct structural templates and score-only API.
- Preserve `chain_pair_pae_mean` and `chain_pair_pae_min`: the kit's upstream
  confidence-summary replacement omits them. Leave it disabled until adapted.
- The audited pipeline recipe uses Torch LayerNorm. The kit's Exact reference
  uses fast LayerNorm: assess that transition separately, not as assumed equality.
- The stock-file installer gate rejects modified upstream files. Build a proper
  fork integration with pinned, reviewed components; do not bypass the gate with
  a force flag and claim supported equivalence.
- Adapt path-dependent helpers to the editable installation. The kit's cache
  retention classifier does not recognize the audited container checkout path.
- Start without near-zero-module skipping (`deadskip`); finite random probes are
  weaker than a general equivalence argument. Revisit only if measured value
  justifies stronger validation.
- Treat A30 and A100 as separate validation targets. Some exact kernel paths are
  H100-specific; the audited sm80 fused triangle-attention path starts at 400 tokens.
- Validate the three-sample pipeline setting explicitly; five-sample benchmarks
  or supported kernel cells do not establish its speed or exactness.

The packaged kit uses CUDA 13 and a newer Torch/cuEquivariance stack. Inventory
host driver versions before building that image. Selective Python changes do not
require that migration. Benchmark an unoptimized fork on any proposed new stack
as a separate control so dependency changes are not confused with kernel gains.

## Stage 4: evaluate Fast for production throughput

Test Fast after Exact integration is stable. It is a credible route to larger
gains, supported by external accuracy evidence. Evaluate fused triangle/MSA
operations, reduced-precision diffusion/atom attention and merged projections
incrementally, then as a combined mode. Do not reduce seeds, recycles, diffusion
steps or samples to improve the comparison.

Numerical differences alone do not imply worse folding. However, matching mean
iPTM alone is insufficient: measure structural quality, candidate selection and
calibration. Fast stays opt-in until the application-specific evidence supports
promotion. Big and multi-GPU sharding are capacity options, not the first choice
for throughput of many independent nanobody complexes.

## Benchmark and release protocol

1. Record actual deployed fork/image digests, driver, GPU model/VRAM, dependencies,
   checkpoint hash and complete effective configuration. Verify the imported
   source path; a PR on this fork does not update a separately pinned container.
2. Freeze a representative panel before tuning: roughly 24 initial complexes,
   stratified by token length (including below/above 400 and 448), VHH MSA depth,
   template arrangement, confident successes and near-threshold failures. Include
   distinct same-sized inputs consecutively to expose stale caches. Include a
   larger supported input to exercise memory limits. Reserve independent targets
   for final validation; this initial panel is an engineering screen, not a
   statistically powered proof of non-inferiority.
3. Use identical input/MSA/template files and a fixed five-seed list for the first
   paired screen, with the production 10 cycles, 200 steps and three samples.
   Repeat unchanged baseline runs to measure natural nondeterminism. Validate
   final candidates for promotion at the normal production seed budget too.
4. Measure total job time and separate startup, featurization, trunk, diffusion,
   confidence and output costs. Report cold and warm performance, per-case paired
   ratios, median/tail runtime, failures, GPU-hours, allocated/reserved memory and
   device-level peak memory. Alternate baseline/optimized run order on exclusive
   GPUs. Report A30 and A100 independently; include warm-up/compilation costs in
   campaign-level accounting. Do not infer end-to-end gains from kernel timings.
5. For overhead/Exact changes compare unrounded coordinates and confidence tensors,
   all summary keys, saved outputs and selected seed/sample IDs. Use matching
   deterministic controls where supported, and separately exercise the actual
   production graph/kernel paths; a deterministic test that disables a path
   does not validate that path.
6. For Fast compare DockQ against experimental structures where available. For
   designed complexes without ground truth, measure scDockQ/pose RMSD, interface
   preservation, iPTM/PAE, selected structures and pipeline pass/fail changes;
   explicitly label these as consistency/proxy metrics, not biological accuracy.
   Use paired target-level confidence intervals and inspect threshold-crossing
   losses/gains. Choose the acceptable non-inferiority criterion before viewing
   results; if uncertainty is too wide for the accuracy requirement, keep Fast
   experimental and expand the panel.
7. Promote one stage at a time only when its intended paths run, compatibility
   checks pass, memory/failure rates remain acceptable and net workload speed
   improves. Keep the original image available for rollback and record execution
   mode in run provenance so recovery cannot silently mix implementations.

## Deployment and current status

This patch has no new dependencies and no pipeline/report schema changes. Updating
an immutable installation requires rebuilding or repinning the Protenix component
image containing these source files, then running its existing GPU smoke test.
The audited pipeline recipe points to a separate source URL/pin: confirm the image
actually consumes this fork's changes before claiming deployment. No core-image
rebuild is intrinsically required by these two edits.

No GPU benchmark, folding-equivalence claim, runtime speedup, production deployment
or merge is established by the CPU checks. Stages 2-4 remain planned work.
