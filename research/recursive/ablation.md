# Recurrent Ablation

## Decision

**No-go:** retain TinyGPT as the control; do not promote single-state recurrence or dual-state refinement. The registered rule required per-seed gains of at least 5 percentage points on graph extrapolation and graph-to-chain extrapolation in at least two of three seeds, with held-out text BPB regression no worse than 0.25. The validation-selected dual-state K=4 model gained 0 points on both reasoning slices in every seed. All generated answer exact-match rates were zero on the measured task slices.

This rejects promotion under this run's small budget. It does not establish that recurrence cannot help with larger training budgets or broader data.

## Protocol

All five generated task families used the fixed Patch 2 instances. Models trained only on the 12 graph training rows, with 4 AdamW updates, batch size 2, target-only loss, and seeds 11, 23, and 37. Loop and dual models were trained at K=4 with weights `[0.1, 0.2, 0.3, 0.4]`; the final-only comparison used the validation-selected dual variant. Selection used graph validation byte NLL only. Evaluation used all four examples per family per interpolation/extrapolation slice and the separate 415-byte text test split. The copy strings were project-authored.

The report contains each seed's task accuracy, confidence interval, error category and latency; held-out NLL/BPB; stage losses; parameter and checkpoint sizes; checkpoint hashes; train/eval split hashes; and resource snapshots. FLOPs were not measured. Block applications are the compute proxy. RSS is the run-level process high-water mark, not an isolated per-condition peak.

## Results

| Condition | Parameters | Unique blocks | Applications per update | Mean train time | Held-out task exact match | Text BPB at reported K |
|---|---:|---:|---:|---:|---:|---:|
| TinyGPT, one pass | 37,344 | 1 | 1 | 0.031 s | 0% | 7.9731 |
| Shared single state, K=4 trained | 37,792 | 1 | 4 | 0.114 s | 0% | 7.9752 |
| Dual state, K=4 trained | 38,624 | 1 | 8 | 0.156 s | 0% | 7.9750 |
| Independent stack, depth 4 | 75,456 | 4 | 4 | 0.080 s | 0% | 7.9808 |
| Dual state, final-only, K=4 | 38,624 | 1 | 8 | 0.160 s | 0% | 7.9750 |

All architectures were below the one-million parameter limit. The independent stack used about twice the parameters of TinyGPT. The dual refiner used twice the block applications per update as the single-state loop at K=4. Mean training examples per second were 264 (TinyGPT), 73 (single state), 51 (dual state), and 101 (depth-4 stack). The complete three-seed run took 93.42 seconds and peaked at 377,257,984 bytes RSS.

Increasing inference K from 1 to 2 to 4 did not change exact-match accuracy on the scored slices. Held-out text BPB remained close to the plain control; the worst mean difference among the principal candidates was below the registered 0.25 BPB regression limit. Copy accuracy was 0% for every condition. Training through multiple refinements did not improve task exact match over final-only loss in this run.

## Limits

Each seed saw only eight sampled training examples (four updates times batch two); each per-seed evaluation slice had four examples, so intervals remain broad. The task generator and text fixture are synthetic/project-authored. Results are not natural-language reasoning, interaction quality, coding ability, or general capability evidence. The exact-match no-go is specific to this training and evaluation budget. K=8 was neither trained nor evaluated. No external data, model, service, GPU, or new runtime dependency was used.

Reproduction: `timeout 120s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /tmp/recurrent-audit-venv/bin/python -m research.recursive.ablate`

Full per-seed results and hashes: [`out.json`](out.json).
