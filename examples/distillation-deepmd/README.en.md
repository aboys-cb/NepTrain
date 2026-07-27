<div align="center">
<strong>English</strong> | <a href="README.md">简体中文</a>
</div>

# Distill a NEP student from a DPA-3 teacher

This tutorial starts from a public DPA-3 model, runs standalone labeling and
student smoke training, and then runs one complete workflow generation:

```text
candidates → DPA-3 teacher → energy/forces/virial
           → TorchNEP student → GPUMD sampling → relabeling and retraining
```

The main path uses DeePMD-kit's built-in `DPA-3.2-5M` with the `OMol25` head.
DPA-4 uses the same NepTrain runner but requires a different model preparation
path described below.

## 1. Enter the example and create an isolated environment

```bash
cd examples/distillation-deepmd
pip install 'NepTrain[deepmd,torchnep]'
```

Verify the commands and CUDA runtime:

```bash
command -v neptrain
command -v dp
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

Standalone labeling can use CPU. Training, teacher labeling, and GPUMD in the
supplied workflow use CUDA. Install a PyTorch/CUDA build matching the machine
if CUDA is unavailable.

## 2. Download and inspect the teacher

```bash
python download_model.py
dp --pt show DPA-3.2-5M.pt model-branch type-map
```

The script uses `dp pretrained download`, copies the pinned model to
`DPA-3.2-5M.pt`, and prints its SHA256. The `dp --pt show` output must include
the `OMol25` branch and H/O elements.

NepTrain takes a local model file instead of a version-dependent alias and
records that file's SHA256 in every labeling task.

## 3. Generate three candidate structures

```bash
python make_candidates.py
```

This writes three periodic water structures with slightly scaled O–H bonds:

```bash
python - <<'PY'
from ase.io import read
frames = read("candidates.xyz", index=":")
print("frames:", len(frames))
print("symbols:", frames[0].get_chemical_formula())
print("cell volume:", frames[0].get_volume())
PY
```

The teacher runner requires a positive cell volume because it converts ASE
stress to virial.

## 4. Label with DPA-3

```bash
neptrain label candidates.xyz \
  --backend model \
  --model DPA-3.2-5M.pt \
  --model-name dpa-3.2-5m-omol25 \
  --runner 'neptrain model-worker deepmd --head OMol25' \
  --device cuda \
  --precision float32 \
  --structures-per-job 2 \
  --wait \
  --output labeled.xyz
```

`--head OMol25` is a DeepMD-runner option and must remain inside the quoted
runner string. NepTrain appends the shared `--model`, `--input`, `--output`,
`--device`, and `--precision` arguments.

Check the result:

```bash
python inspect_labels.py
```

The last line must start with `OK:` and show one teacher SHA256 for all three
frames. Every frame must have finite energy, forces, and a `(3, 3)` virial.

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

`nep-smoke.in` trains for only 10 epochs. It verifies that TorchNEP can consume
DeepMD labels and publish `nep.txt`; this model is not suitable for MD.

```bash
test -s student-nep.txt
test -s student-train/training-report.json
test -s student-train/training-convergence.png
```

Training plots are PNG only.

## 6. Run one complete workflow generation

The full path also needs GPUMD:

```bash
command -v gpumd
neptrain doctor --project project.yaml
```

The three-atom test uses `gpumd-nve.in`. It initializes velocities at 50 K but
does not apply a thermostat, avoiding strong thermostat coupling in a tiny
system. `nep-workflow.in` trains for 500 epochs before this short MD run.

```bash
neptrain workflow run project.yaml --prepare-only
neptrain workflow run deepmd-distillation-workflow
neptrain workflow status deepmd-distillation-workflow --jobs
```

To observe the controller in the current terminal:

```bash
neptrain workflow run project.yaml --foreground
```

## 7. Inspect the result

| Location | Contents |
|---|---|
| `generations/0001/train/` | Initial student and PNG convergence plot |
| `generations/0001/md/` | GPUMD trajectory and `trajectory-health.json` |
| `generations/0001/select/` | FPS-selected candidates |
| `generations/0001/label/selected-labels.xyz` | New DPA-3 labels |
| `generations/0001/label/label-provenance.json` | Runner, head, model name, and SHA256 |
| `generations/0001/dataset/` | Merged student training set |
| `generations/0001/retrain/` | Retrained student |
| `generations/0001/evaluate/` | Evaluation and activation artifacts |

`budget_exhausted` means the one-generation tutorial budget was consumed, not
that the workflow failed. All eight stages should complete; teacher or MD
failures are shown on their respective stage.

## 8. Switch to DPA-4

DPA-4/SeZM is PyTorch-only in DeePMD-kit and uses `.pt2` after freeze. Install
a DeePMD-kit version with DPA-4 support and freeze/export on the target GPU:

```bash
pip install --pre 'deepmd-kit[torch]>=3.2.0b0,<4'
dp --pt freeze -c model.ckpt.pt -o frozen-dpa4
```

Use the same runner:

```bash
neptrain label candidates.xyz \
  --backend model \
  --model frozen-dpa4.pt2 \
  --model-name dpa4-sezm \
  --runner 'neptrain model-worker deepmd' \
  --device cuda \
  --precision float32 \
  --wait \
  --output labeled-dpa4.xyz
```

Add `--head <name>` inside the runner for a multitask model. A `.pt2` contains
artifacts tied to its freeze environment, so smoke-test it on the deployment
GPU before putting it into a workflow. The repository has run the DPA-3
workflow end to end; this DPA-4 section documents the shared interface and does
not claim validation against a public DPA-4 checkpoint.

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `dp: command not found` | DeepMD extra is in another environment | Reinstall and confirm the active conda environment |
| `head ... not found` | The model does not contain that branch | Run `dp --pt show ... model-branch` |
| `cannot access a CUDA device` | The process has no GPU allocation | Check Slurm GPU resources and PyTorch CUDA |
| `returned non-finite labels` | Teacher inference failed | Test the same model and structure with `dp test` or ASE |
| GPUMD produces NaN | Student or MD settings are unstable | Inspect the health report; keep the tutorial NVE before changing ensembles |
| Spin input is rejected | Runner does not produce `mforce` | Use a teacher backend that supports magnetic-force labels |

References: [built-in model downloads](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/pretrained.html),
[DPA-3 distillation tutorial](https://docs.deepmodeling.com/projects/deepmd/en/latest/getting-started/dpa3_cyclohexane_distillation.html),
and [DPA-4 documentation](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dpa4.html).
