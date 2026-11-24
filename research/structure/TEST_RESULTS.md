# Regression Test Record

Baseline was run before source edits on 2026-11-24 using Python 3.14.2 and
PyTorch 2.15.0+cpu from `/tmp/audit-pytorch-venv`. The isolated environment
was created from the repository's existing
`research/experiments/requirements_cpu.lock.txt`; no repository dependency
files were edited.

Command:

```sh
timeout 180s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/audit-pytorch-venv/bin/python -m unittest -v \
  research.foundation.test research.interaction.test research.orchestrator.test \
  research.adaptive.test research.recursive.test
```

Baseline result: **51 tests passed**, 13.042 seconds.

The command includes Foundation architecture and exact checkpoint resume,
interaction corpus and experience handling, orchestrator controls, adaptive
symbolic strategy extraction and transfer, and recursive transfer/training
regressions. The post-edit run, after adding the episode sampler replay test,
passed **52 tests**, 11.444 seconds. The same command and environment were
used. No test output was redirected or edited into the results.

## Structural Environment And Training

Command:

```sh
timeout 180s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/audit-pytorch-venv/bin/python -m unittest -v research.structure.test
```

Final result after the v5 data observability, answer-objective, and generation
metric changes: **10 tests passed**, 9.075 seconds. Coverage includes deterministic
oracle generation, world/split/template isolation, complete structural facts
in language surfaces, direction checks, cross-split wording checks, context
limits, all three objective paths, resource guards, and exact
interrupted/resumed checkpoint equivalence.

All three 2-update, full-model smoke arms completed with checkpoint and
evaluation reports. A separate full-model A run interrupted at update one and
resumed to update two matched the uninterrupted model, optimizer, scheduler,
sampler, history, step, and Python/Torch RNG states exactly. The 12-update
bounded pilots for A/B/C completed and are recorded in
[`training_manifest_v5.json`](training_manifest_v5.json).
