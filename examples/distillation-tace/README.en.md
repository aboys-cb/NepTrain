<div align="center">
<strong>English</strong> | <a href="README.md">简体中文</a>
</div>

# Distill a NEP student from a TACE teacher

This tutorial labels three Al crystals with a pinned `TACE-OAM-7M` model,
trains a TorchNEP student, and can then run one complete workflow generation:

```text
candidates → TACE teacher → energy/forces/virial
           → TorchNEP student → GPUMD sampling → relabeling and retraining
```

## 1. Enter the example and install the environment

TACE is currently installed from its official repository. This tutorial pins
the commit whose `tace-eval` interface was checked:

```bash
cd examples/distillation-tace
pip install torch
pip install 'NepTrain[torchnep]'
pip install \
  'TACE[cueq12] @ git+https://github.com/xvzemin/tace.git@4b977dcc13ee87d8ba6cceba3ffb7abe43c087c8'
export TACE_USE_CUE=1
```

Verify that both commands come from the same environment:

```bash
command -v neptrain
command -v tace-eval
python -c 'import torch, tace; print(torch.__version__, torch.cuda.is_available())'
```

Standalone labeling can run on CPU. The complete workflow uses CUDA and GPUMD.
The `cueq12` extra installs the CUDA 12 cuEquivariance operators. Its official
wheel requires an Ampere-or-newer GPU; it failed on a Sai V100 with
`cudaErrorNoKernelImageForDevice`. On V100, install TACE without `cueq12` and
leave `TACE_USE_CUE=0`.

## 2. Download the pinned teacher

```bash
python download_model.py
```

The script downloads `TACE-OAM-7M.pt` from a pinned revision of the official
TACE Hugging Face repository and verifies a fixed SHA256. NepTrain records the
same local model content in label provenance.

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

## 4. Label with TACE

Run the interface on CPU first:

```bash
neptrain label candidates.xyz \
  --backend model \
  --model TACE-OAM-7M.pt \
  --model-name TACE-OAM-7M \
  --runner 'neptrain model-worker tace --fidelity-index 0' \
  --device cpu \
  --precision float32 \
  --structures-per-job 2 \
  --wait \
  --output labeled.xyz
```

Upstream `tace-eval` writes a prediction extxyz in a temporary directory.
NepTrain checks frame count, finite values, and label shapes, then normalizes
`TACE_energy`, `TACE_forces`, and `TACE_stress`/`TACE_virials` into canonical
`energy`, `forces`, and `virial` fields before publication.

```bash
python inspect_labels.py
```

The final line must start with `OK:`. All frames must share one teacher SHA256
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
  --seed 20260728 \
  --workdir student-train \
  --output student-nep.txt
```

The ten epochs in `nep-smoke.in` only prove that TorchNEP can consume the
teacher labels. The resulting model has no scientific value.

```bash
test -s student-nep.txt
test -s student-train/training-report.json
test -s student-train/training-convergence.png
```

## 6. Run one complete workflow generation

```bash
command -v gpumd
neptrain doctor --project project.yaml
neptrain workflow run project.yaml --prepare-only
neptrain workflow run tace-distillation-workflow
neptrain workflow status tace-distillation-workflow --jobs
```

Use `neptrain workflow run project.yaml --foreground` to observe the controller
in the current terminal. A final `budget_exhausted` means the one-generation
tutorial budget was consumed; it is not a failure.

## 7. Inspect the result

| Location | Contents |
|---|---|
| `generations/0001/train/` | Initial student and PNG convergence plot |
| `generations/0001/md/` | GPUMD trajectory and health report |
| `generations/0001/select/` | FPS selection result |
| `generations/0001/label/selected-labels.xyz` | New TACE labels |
| `generations/0001/label/label-provenance.json` | Runner, model name, and SHA256 |
| `generations/0001/dataset/` | Merged training set |
| `generations/0001/retrain/` | Retrained student |
| `generations/0001/evaluate/` | Activation result |

## 8. Spin-teacher boundary

TACE supports models with noncollinear magnetic forces, but the ordinary
`TACE-OAM-7M` model is not automatically a magnetic teacher. NepTrain accepts
spin input only when the checkpoint actually emits
`noncollinear_magnetic_forces`, which are mapped to canonical `mforce`.
Missing magnetic-force output fails explicitly and is never replaced by zeros.

## Before production use

- Match the checkpoint elements, theory level, and license to the target.
- Do not mix teacher labels from different fidelities or energy references.
- Use an independent validation set and physically meaningful thresholds.
- Increase student capacity, epochs, candidate coverage, and workflow budget.
- Smoke-test the exact checkpoint on the deployment GPU before using Slurm.

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `tace-eval is not installed` | TACE is in another environment | Check `command -v tace-eval` |
| `model SHA256 mismatch` | Incomplete or changed download | Delete the model/`.part` file and retry |
| `returned neither virial nor stress` | The checkpoint lacks the required head | Use a model with all training labels |
| CUDA OOM | Model or shard is too large | Reduce `structures_per_job`; test on CPU |
| Spin input lacks magnetic forces | The checkpoint is not a spin teacher | Use a TACE model trained for noncollinear magnetic forces |

See the [TACE repository](https://github.com/xvzemin/tace) and
[TACE inference documentation](https://tace.readthedocs.io/en/latest/guide/scripts.html).
