<div align="center">
<strong>English</strong> | <a href="README.md">简体中文</a>
</div>

# Distill a NEP student from a MACE teacher

This tutorial labels three Al structures with a pinned MACE-MP-0 checkpoint,
trains a TorchNEP student, and can then run one complete workflow generation:

```text
candidates → MACE teacher → energy/forces/virial
           → TorchNEP student → GPUMD sampling → relabeling and retraining
```

## 1. Enter the example and create the environment

```bash
cd examples/distillation-mace
pip install 'NepTrain[mace,torchnep]'
```

Install a PyTorch build suitable for your driver first, then verify:

```bash
command -v neptrain
python -c 'import torch, mace; print(torch.__version__, torch.cuda.is_available())'
```

Standalone labeling can run on CPU. The complete `project.yaml` workflow uses
CUDA and GPUMD.

## 2. Download the pinned teacher

```bash
python download_model.py
```

The script downloads the MACE-MP-0 128-channel small checkpoint to
`mace-mp-0-small.model` and verifies the SHA256 pinned in the script. Existing
files are verified again to avoid silently using different weights.

This example deliberately avoids the dynamic default of `mace_mp()`, which may
change across MACE releases. NepTrain records the local checkpoint SHA256 in
label provenance.

## 3. Generate candidates

```bash
python make_candidates.py
```

`candidates.xyz` contains three fcc Al cells with different lattice scaling:

```bash
python - <<'PY'
from ase.io import read
frames = read("candidates.xyz", index=":")
print("frames:", len(frames))
print("atoms per frame:", len(frames[0]))
print("volumes:", [round(frame.get_volume(), 3) for frame in frames])
PY
```

## 4. Label with MACE

Run the interface on CPU first:

```bash
neptrain label candidates.xyz \
  --backend model \
  --model mace-mp-0-small.model \
  --model-name mace-mp-0-small \
  --runner 'neptrain model-worker mace' \
  --device cpu \
  --precision float32 \
  --structures-per-job 2 \
  --wait \
  --output labeled.xyz
```

Use `--device cuda` after the CPU path works. Check the labels:

```bash
python inspect_labels.py
```

The last line must start with `OK:`. All frames must share one teacher SHA256
and contain finite energy, forces, and a `(3, 3)` virial.

## 5. Train a smoke-test student

```bash
neptrain train labeled.xyz \
  --backend torchnep \
  --config nep-smoke.in \
  --device cuda \
  --torch-backend bmm \
  --precision float32 \
  --no-compile \
  --seed 20260727 \
  --workdir student-train \
  --output student-nep.txt
```

`nep-smoke.in` trains for only 10 epochs. It proves that TorchNEP can consume
the teacher labels and publish `nep.txt`; the model has no scientific value.

```bash
test -s student-nep.txt
test -s student-train/training-report.json
test -s student-train/training-convergence.png
```

Training plots are PNG only.

## 6. Run one complete workflow generation

Check GPUMD and the project:

```bash
command -v gpumd
neptrain doctor --project project.yaml
```

The project uses local process targets and is intended for an already allocated
GPU node. `nep-workflow.in` trains for 500 epochs and `gpumd-nve.in` runs a
very short NVE sample.

```bash
neptrain workflow run project.yaml --prepare-only
neptrain workflow run mace-distillation-workflow
neptrain workflow status mace-distillation-workflow --jobs
```

Or run the controller in the foreground:

```bash
neptrain workflow run project.yaml --foreground
```

## 7. Inspect the result

| Location | Contents |
|---|---|
| `generations/0001/train/` | Initial student and PNG convergence plot |
| `generations/0001/md/` | GPUMD trajectories and health report |
| `generations/0001/select/` | FPS selection result |
| `generations/0001/label/selected-labels.xyz` | New MACE labels |
| `generations/0001/label/label-provenance.json` | Runner, model name, and SHA256 |
| `generations/0001/dataset/` | Merged training set |
| `generations/0001/retrain/` | Retrained student |
| `generations/0001/evaluate/` | Evaluation result |

`budget_exhausted` means the one-generation tutorial budget was consumed; it
does not mean the workflow failed.

## 8. Before production use

- Choose a checkpoint whose elements, theoretical level, and license match the
  target system.
- Cover the intended thermodynamic and structural space, not three scaled cells.
- Prepare an independent validation set and physically meaningful thresholds.
- Increase student capacity, training epochs, and workflow generations.
- Replace local targets with cluster-appropriate Slurm targets.
- Never mix teacher labels with different energy references or theory levels.

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `model SHA256 mismatch` | Incomplete or changed download | Delete the model/`.part` file and download again |
| `MACE is not installed` | Extra installed in another environment | Check `which python` and `pip show mace-torch` |
| CUDA OOM | Checkpoint or shard is too large | Reduce `structures_per_job`; smoke-test on CPU |
| `returned non-finite labels` | Teacher is unsuitable or input is invalid | Test the same structure with the MACE ASE calculator |
| Spin input is rejected | MACE runner does not produce `mforce` | Use a teacher backend that supports magnetic-force labels |

See [MACE Foundation Models](https://mace-docs.readthedocs.io/en/latest/guide/foundation_models.html).
