<div align="center">
<strong>English</strong> | <a href="README.md">简体中文</a>
</div>

# Run one NepTrain workflow generation from ABACUS

This tutorial is for first-time NepTrain users on Slurm. It exercises:

```text
seed data → student training → GPUMD sampling → FPS → ABACUS labeling
          → pre-training accuracy assessment → dataset merge → next generation or final training
```

The example uses Al with an ABACUS plane-wave basis. The repository does not
distribute ABACUS pseudopotentials; you need a working `abacus` executable and
the corresponding UPF file.

## 1. Enter the example and install the environment

```bash
cd examples/workflow-abacus-slurm
python -m pip install -e '../..[torchnep]'
```

Install from this checkout so NepTrain matches the supplied placeholder templates. Use the same source revision on MD worker nodes. The `gpumd-npt.in` file is rendered by NepTrain; do not pass it directly to GPUMD.

Training nodes need TorchNEP, MD nodes need GPUMD, and labeling nodes need
ABACUS. Put their conda/module setup in `env-training.sh`, `env-gpumd.sh`, and
`env-abacus.sh`.

## 2. Generate unlabeled seed structures

```bash
python ../prepare_al_seed.py --output-dir .
```

This generates `seed-train.xyz` (24 frames), `seed-validation.xyz` (4 optional test frames),
and `structures/al.xyz`. It creates no energy, force, virial, or EMT labels. Step 6 labels the
seeds with the same backend, inputs, and resource manifest used by the workflow.

## 3. Pin ABACUS resources

Copy the manifest template:

```bash
cp abacus-resources.example.json abacus-resources.json
```

For resource root `/shared/abacus-resources` and file
`Al_ONCV_PBE.upf`, obtain its real hash:

```bash
sha256sum /shared/abacus-resources/Al_ONCV_PBE.upf
grep -im1 "element" /shared/abacus-resources/Al_ONCV_PBE.upf
```

On macOS, use `shasum -a 256`. Update `abacus-resources.json`:

```json
{
  "elements": {
    "Al": {
      "orbital": null,
      "pseudopotential": {
        "path": "Al_ONCV_PBE.upf",
        "sha256": "<64-character lowercase hash>"
      }
    }
  },
  "protocol": "neptrain.abacus-resources.v1",
  "release": "<your resource release>"
}
```

The supplied `INPUT` uses `basis_type pw`, so `orbital` may be `null`. With
`basis_type lcao`, provide the `.orb` relative path and SHA256 or NepTrain will
reject the task before starting ABACUS.

## 4. Configure the cluster

In `project.yaml`, replace:

1. `REPLACE_GPU_PARTITION` with the training and GPUMD partition.
2. `REPLACE_CPU_PARTITION` with the ABACUS partition.
3. Both `/REPLACE/WITH/YOUR/abacus-resources` values with the absolute resource
   root visible from login and compute nodes.
4. Placeholder conda/module commands in the three `env-*.sh` files.

The example uses `NEPTRAIN_ABACUS_COMMAND: srun abacus`. Change that
environment variable if your cluster uses another launcher.

The supplied `INPUT` contains `kspacing 0.25`, so `kpoint_mode: auto` keeps it.
Continue only when this command has no output:

```bash
grep -R "REPLACE" project.yaml env-*.sh abacus-resources.json
```

## 5. Check the runtime prerequisites

Finish replacing resource paths, hashes, partitions, and setup commands before submitting
any job. The full `doctor --project` input check runs after the seed labeling below;
`train.xyz` does not exist yet. The next single-structure calculation tests the actual backend.

## 6. Label one structure first

```bash
neptrain label structures/al.xyz \
  --backend abacus \
  --project project.yaml \
  --target abacus \
  --wait \
  --output abacus-check.xyz
```

Check all required labels:

```bash
python - <<'PY'
from ase.io import read
atoms = read("abacus-check.xyz")
print("energy:", atoms.get_potential_energy())
print("max |force|:", abs(atoms.get_forces()).max())
print("virial shape:", atoms.info["virial"].shape)
PY
```

On failure, inspect `neptrain task logs <run_directory>`. Do not start the
workflow until standalone labeling succeeds.

Once the single-structure check passes, label the seeds with the same settings. Only the training set is required; an absent optional test set produces a warning and is skipped.

```bash
neptrain label seed-train.xyz --backend abacus --project project.yaml --target abacus --wait --output train.xyz
# Optional / 可选：
neptrain label seed-validation.xyz --backend abacus --project project.yaml --target abacus --wait --output validation.xyz
neptrain doctor --project project.yaml
```

## 7. Prepare and start the workflow

```bash
neptrain workflow run project.yaml --prepare-only
neptrain workflow run abacus-tutorial-workflow
neptrain workflow status abacus-tutorial-workflow --jobs
```

To stop the controller and cancel its current jobs:

```bash
neptrain workflow stop abacus-tutorial-workflow
```

## 8. Inspect the completed generation

| Location | Contents |
|---|---|
| `generations/0001/explore/` | GPUMD trajectories and health report |
| `generations/0001/select/` | FPS results and `selection-pca.png` |
| `generations/0001/label/selected-labels.xyz` | New ABACUS labels |
| `generations/0001/label/label-provenance.json` | INPUT, UPF/ORB hashes, and backend provenance |
| `generations/0001/update/` | Merged training set |
| `generations/0001/train/` | Sampling model, `training-convergence.png`, and `training-parity-train.png` |
| `generations/0001/evaluate/` | Prediction error on new labels before training on them, plus `acquisition-parity.png` |

The `evaluation:` configuration section controls optional diagnostics only and does not block progression. Acquisition accuracy thresholds live in
`workflow.convergence`; the example values are for the tutorial only. Final training requires accuracy,
production coverage, and the configured passing streak. Optional test reports are in `generations/0001/train/`.

A final `budget_exhausted` state means the example used its one-generation
budget. Determine real failures from stage/job states and logs.

## 9. Convert it to a production project

- Expand the training set with labels from the same ABACUS theoretical level.
- Pin UPF files for every element and ORB files for LCAO.
- Review `ecutwfc`, k points, smearing, magnetism, and SCF convergence.
- Replace the 10–80-step NPT path with validated sampling.
- Increase data volume, validation coverage, training size, and generations.
- Tune Slurm resources and labeling concurrency.

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `resource hash mismatch` | UPF/ORB differs from the manifest | Recompute SHA256 |
| `does not declare its element` | UPF lacks recognizable `element=` metadata | Use the correct ABACUS UPF and inspect it |
| `missing orbital` | LCAO basis without an ORB manifest record | Add `.orb` path and hash |
| `execution target ... FAIL` | Placeholder setup or partition remains | Remove every `REPLACE` value |
| ABACUS exits immediately | Module, MPI launcher, or INPUT mismatch | Inspect task/stage logs |
| NaN trajectory | Unstable student or MD settings | Inspect the health report; reduce timestep/temperature and improve seed data |

## NPT settings and validation limits

`gpumd-npt.in` uses isotropic `npt_scr`, target pressure 0 GPa, a rough elastic-modulus
parameter of 100 GPa, and thermostat/barostat coupling parameters of 100/1000 steps.
These are tutorial settings, not fitted material constants. The four-atom cell and 10–80
steps test file and stage handling; they cannot establish equilibration or a production model.
See the [GPUMD ensemble syntax](https://gpumd.org/gpumd/input_parameters/ensemble_standard.html).
The one-generation budget normally ends as `budget_exhausted`, not `complete`.
