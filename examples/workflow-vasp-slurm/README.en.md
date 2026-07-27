<div align="center">
<strong>English</strong> | <a href="README.md">简体中文</a>
</div>

# Run one NepTrain workflow generation from VASP

This tutorial is for first-time NepTrain users on Slurm. It exercises a real:

```text
seed data → student training → GPUMD sampling → FPS → VASP labeling
          → dataset merge → student retraining → evaluation
```

The example uses Al and VASP for real labels on newly selected structures. The
repository cannot distribute VASP or POTCAR files. You need a working
`vasp_std` and licensed PAW_PBE resources.

## 1. Enter the example and install the environment

From the repository root:

```bash
cd examples/workflow-vasp-slurm
pip install 'NepTrain[torchnep]'
```

Training nodes must run `neptrain` and TorchNEP, MD nodes must run `gpumd`, and
labeling nodes must run `vasp_std`. Record the conda/module setup for each node
type and put it in `env-training.sh`, `env-gpumd.sh`, and `env-vasp.sh`.

## 2. Generate tutorial seed data

```bash
python ../prepare_al_seed.py --output-dir .
```

The script creates:

| File | Purpose |
|---|---|
| `train.xyz` | 24 seed structures with energy, forces, and virial |
| `validation.xyz` | 4 independent tutorial evaluation structures |
| `structures/al.xyz` | Initial structure for GPUMD |

These labels come from ASE EMT and only validate workflow mechanics. For
production, replace both datasets with data consistent with your VASP setup.
Never mix different theoretical levels in one training set.

## 3. Pin the POTCAR

Copy the manifest template:

```bash
cp vasp-resources.example.json vasp-resources.json
```

For a resource root `/shared/potpaw_PBE` and
`/shared/potpaw_PBE/Al/POTCAR`, obtain the real hash and `TITEL`:

```bash
sha256sum /shared/potpaw_PBE/Al/POTCAR
grep -m1 TITEL /shared/potpaw_PBE/Al/POTCAR
```

On macOS, use `shasum -a 256` instead. Write the results to
`vasp-resources.json`:

```json
{
  "elements": {
    "Al": {
      "path": "Al/POTCAR",
      "sha256": "<64-character lowercase hash>",
      "titel": "<complete value to the right of TITEL =>"
    }
  },
  "family": "PAW_PBE",
  "protocol": "neptrain.vasp-resources.v1",
  "release": "<your POTCAR release>"
}
```

`path` must be `Al/POTCAR` or `Al_<setup>/POTCAR`. NepTrain uses the same
record to validate the file and choose the ASE setup, preventing a different
POTCAR from being used after validation.

## 4. Configure the cluster

Open `project.yaml` and replace:

1. `REPLACE_GPU_PARTITION` with the training and GPUMD partition.
2. `REPLACE_CPU_PARTITION` with the VASP partition.
3. Both `/REPLACE/WITH/YOUR/potpaw_PBE` values with the absolute POTCAR root
   seen from login and compute nodes.
4. The placeholder conda/module commands in all three `env-*.sh` files.

The example launches VASP with `NEPTRAIN_VASP_COMMAND: srun vasp_std`. Change
only that value if your cluster uses another launcher.

The supplied `INCAR` contains `KSPACING = 0.25` and `KGAMMA = True`, so
`kpoint_mode: auto` preserves these settings. Continue only when this produces
no output:

```bash
grep -R "REPLACE" project.yaml env-*.sh vasp-resources.json
```

## 5. Run preflight checks

```bash
neptrain doctor --project project.yaml
```

This checks the schema-v8 project, Slurm commands, setup scripts, and POTCAR
path and SHA256 on compute nodes. It cannot replace one real VASP run, which
must validate the license, MPI setup, and executable.

## 6. Label one structure first

Before starting the workflow, run a real backend smoke test:

```bash
neptrain label structures/al.xyz \
  --backend vasp \
  --project project.yaml \
  --target vasp \
  --wait \
  --output vasp-check.xyz
```

Verify all required labels:

```bash
python - <<'PY'
from ase.io import read
atoms = read("vasp-check.xyz")
print("energy:", atoms.get_potential_energy())
print("max |force|:", abs(atoms.get_forces()).max())
print("virial shape:", atoms.info["virial"].shape)
PY
```

If it fails, inspect `neptrain task logs <run_directory>` before proceeding.

## 7. Prepare and start the workflow

Create the work directory first so you can inspect the resolved configuration:

```bash
neptrain workflow run project.yaml --prepare-only
neptrain workflow run vasp-tutorial-workflow
neptrain workflow status vasp-tutorial-workflow --jobs
```

To stop the controller and cancel its currently managed jobs:

```bash
neptrain workflow stop vasp-tutorial-workflow
```

## 8. Inspect the completed generation

| Location | Contents |
|---|---|
| `generations/0001/md/` | GPUMD trajectories and health report |
| `generations/0001/select/` | FPS selections and report |
| `generations/0001/label/selected-labels.xyz` | New VASP labels |
| `generations/0001/label/label-provenance.json` | VASP inputs and resource provenance |
| `generations/0001/dataset/` | Merged training set |
| `generations/0001/retrain/` | Retrained model and PNG convergence plot |
| `generations/0001/evaluate/` | Evaluation result and PNG plots |

The example sets `max_model_generations: 1`. A final `budget_exhausted` state
means the tutorial used its one-generation budget; it is not a Slurm or VASP
failure. Real failures appear in stage/job state and logs.

## 9. Convert it to a production project

- Replace EMT data with labels from the same VASP theoretical level.
- Add every target element to the resource manifest.
- Review `ENCUT`, pseudopotentials, k points, electronic convergence, and spin.
- Replace the 10–80-step NVE path with validated NVE/NVT/NPT sampling.
- Increase data volume, validation coverage, training size, and generations.
- Tune `structures_per_job`, `max_concurrent`, and Slurm resources.

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `resource hash mismatch` | POTCAR changed or manifest is wrong | Recompute SHA256; do not bypass validation |
| `POTCAR version mismatch` | `titel` is incomplete | Copy the complete value from `grep -m1 TITEL` |
| `execution target ... FAIL` | Placeholder setup or partition remains | Remove every `REPLACE` value |
| VASP exits immediately | Module, license, or MPI launcher mismatch | Inspect task/stage logs |
| NaN trajectory | Unstable initial student or MD settings | Inspect `trajectory-health.json`; reduce timestep/temperature and improve seed data |
