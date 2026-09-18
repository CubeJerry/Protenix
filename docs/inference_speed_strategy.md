# Protenix-v2 inference speed strategy

## Implementation status — 18 September 2026

| Stage | Implementation | Validation and release status |
|---|---|---|
| 1. Memory/allocator overhead | Implemented in the fork | 3 CPU regressions passed; GPU memory/runtime checks pending |
| 2. Startup and repeated preparation | All four changes implemented across fork and pipeline | 6 model CPU tests and 5 pipeline shell tests passed; new optimizations remain opt-in |
| 3. Exact execution integration | Planned | Begin after stages 1–2 are validated on target GPUs |
| 4. Fast execution evaluation | Planned | Requires separate application-specific accuracy assessment |

Implementation is available in [Protenix PR #1](https://github.com/CubeJerry/Protenix/pull/1)
and [pipeline PR #122](https://github.com/CubeJerry/dev/pull/122), both published
as drafts. The implementation revisions are fork
`a3b3a3e28783896ad9f9ee17b2cf8fc484405891` and pipeline
`8fc6a29e7586e3ce5832dc5ba2ad023c5bb6ccf9`. This status records implementation
and CPU checks, not deployment or measured folding acceleration.

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

## Stage 1: minimal overhead changes

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

| Implemented change | Completed verification | Remaining target-system verification |
|---|---|---|
| Persistent native Triton/CUDA caches in the SLURM template | Default/override paths, cross-job reuse of paths, namespace separation, unavailable-path fallback and invalid namespaces | Actual cache hits, cold/warm timing, concurrent writes and full-cache behavior |
| Skip overwritten random linear initialization with strict loading | Representative linear checkpoint equality, constant buffers, strict-load rejection and isolated initialization scope | Full Protenix-v2 checkpoint parameters/buffers and seeded end-to-end outputs |
| Reuse identical template calculations within one call | Bitwise CPU output equality, fewer calls, distinct features/geometry masks, changed pair inputs and training/gradient guards | GPU equivalence, duplicate frequency, equality-check overhead and memory |
| Reuse MSA pair weights across chunks within one call | Bitwise CPU output equality, fewer projections, partial final chunks, changed pair inputs and single-chunk/training behavior | GPU equivalence and memory; confirm production VHH MSAs span multiple chunks |

Stage 2 is implemented behind independent opt-in switches (all default to `0`):

```bash
export PROTENIX_SKIP_RANDOM_INIT=1
export PROTENIX_REUSE_TEMPLATES=1
export PROTENIX_REUSE_MSA_PAIR_WEIGHTS=1
```

The runner logs these settings and rejects values other than `0`/`1`.
Initialization skipping is scoped to construction of this model's linear layers;
it never patches PyTorch globally. Zero/one constants and nonpersistent buffers
retain their initialization. Non-strict checkpoint loading is rejected. The
normal runner reseeds after checkpoint loading and before stochastic inference;
custom callers must do the same. Missing checkpoint parameters still fail the
existing strict load. This is selective skipping, not uninitialized construction
of every module, and does not alter checkpoint keys or parameter shapes.

Template reuse compares all five template feature slices and the fork's optional
per-template geometry mask. Only exact duplicates share a computation, only in
eval mode with gradients disabled, and only within one forward call. Every
original slot contributes in its original order. Only outputs needed by later
duplicates are retained, and each is released after its final use. Equality
checks may synchronize a GPU, so workloads without duplicate templates may not
benefit; measure this switch separately.

MSA pair-weight reuse computes the same normalization, projection and softmax
once per MSA stack call when multiple chunks share z. It preserves all MSA rows,
chunk boundaries and downstream operations. Single-chunk, training and gradient
paths retain their original calculation. Nothing is cached across recycles,
candidates or seeds. GPU peak memory remains a required validation item.

The companion pipeline patch supports persistent **native Triton/CUDA caches**:

```bash
export NOMINEE_PROTENIX_JIT_CACHE_ROOT=/persistent/user-writable/protenix-jit
export NOMINEE_PROTENIX_JIT_CACHE_NAMESPACE=imageDigest-forkCommit-a100-driverVersion
```

Use an actual pinned image/fork/GPU/driver identifier in the namespace, and change
it whenever any of those changes. The image identity must capture Torch, CUDA,
Triton and cuEquivariance versions. Paths are separated by numeric user ID and
namespace; existing `TRITON_CACHE_DIR` and `CUDA_CACHE_PATH` overrides win. No
persistent root means the existing task-local policy. Unavailable persistent
directories fall back to task-local caches. A missing/unsafe namespace is a
configuration error. Native backends remain responsible for their cache keys
and writes; this does not enable new kernels, compilation modes or autotuning.
No cache files are copied or rewritten, avoiding path-dependent cache manifests.

Prewarm with one job before launching an array. Verify native backend concurrent
writes on the actual cluster filesystem before broad use; CPU tests exercise
path selection, namespaces, overrides and unavailable-path fallback only. Cache
capacity/exhaustion, driver compatibility and cold/warm GPU timings remain
unverified. This is an opt-in deployment facility, not an assertion that every
backend or filesystem has already passed concurrency testing.

Validation: CPU tests run the actual MSA/template modules with nonzero randomized
parameters and compare outputs with `torch.equal`, check projection/template call
counts, distinct geometry/features, changed pair inputs, training/gradient guards,
strict loading of representative linear modules, initialization scope isolation,
and nonpersistent constant buffers. They do not validate the full checkpoint or
end-to-end folding. The stage-1 memory-lifetime tests continue to pass.

Model reuse across candidates remains a later option only if profiling shows
construction is a material cost; it requires config/cache reset and different
job ownership.

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

## Next work when cluster access returns

1. Build or repin the Protenix component image to the implemented fork revision;
   update the pipeline and regenerate its job scripts. Confirm the imported source
   and logged switches. Keep all stage-2 switches off for the initial control.
2. Compare the pre-stage-1 fork with stage 1, then enable each stage-2 change
   individually before testing their combination. Keep checkpoint, inputs,
   precision and sampling budget fixed throughout.
3. Validate full-checkpoint loading and seeded coordinates/confidence outputs on
   A30 and A100. Include duplicate and distinct templates, per-template geometry
   masks, shallow/deep MSAs and consecutive distinct candidates.
4. Prewarm one persistent-cache job, then check warm reuse and concurrent jobs on
   the actual filesystem. Record startup, total runtime and peak memory alongside
   numerical comparisons using the benchmark protocol above.
5. Enable only changes that pass those checks and improve the real workload.
   Leave ineffective switches off. Proceed to stage 3 once this baseline is stable.

While the login node is unavailable, no GPU timing or full-folding results are
claimed. The 14 passing CPU tests comprise 3 stage-1, 6 stage-2 model and 5 pipeline
tests; they are not 14 end-to-end folding runs.

## Deployment and current status

This patch has no new dependencies and no pipeline/report schema changes. Updating
an immutable installation requires rebuilding or repinning the Protenix component
image containing these source files, then running its existing GPU smoke test.
The audited pipeline recipe points to a separate source URL/pin: confirm the image
actually consumes this fork's changes before claiming deployment. No core-image
rebuild is intrinsically required by these source edits.

No GPU benchmark, folding-equivalence claim, runtime speedup, production deployment
or merge is established by the CPU checks. Stage 2 is implemented but opt-in;
stages 3-4 remain planned work.
