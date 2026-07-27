<div align="center">
<strong>English</strong> | <a href="README.md">简体中文</a>
</div>

# Run one NepTrain workflow generation from ABACUS

This tutorial is for first-time NepTrain users on Slurm. It exercises:

```text
seed data → student training → GPUMD sampling → FPS → ABACUS labeling
          → dataset merge → student retraining → evaluation
```

The example uses Al with an ABACUS plane-wave basis. The repository does not
distribute ABACUS pseudopotentials; you need a working `abacus` executable and
the corresponding UPF file.

## 1. Enter the example and install the environment

```bash
cd examples/workflow-abacus-slurm
pip install 'NepTrain[torchnep]'
```

Training nodes need TorchNEP, MD nodes need GPUMD, and labeling nodes need
ABACUS. Put their conda/module setup in `env-training.sh`, `env-gpumd.sh`, and
`env-abacus.sh`.

## 2. Generate tutorial seed data

```bash
python ../prepare_al_seed.py --output-dir .
```

This creates `train.xyz`, `validation.xyz`, and `structures/al.xyz`. The first
two use ASE EMT labels only to exercise workflow mechanics. Replace them with
first-principles data consistent with your ABACUS pseudopotentials,
functional, basis, and k-point setup before production use.

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

## 5. Run preflight checks

```bash
neptrain doctor --project project.yaml
```

This checks the schema, Slurm target, setup scripts, and UPF path and SHA256 on
compute nodes. It does not run an ABACUS calculation or validate the MPI/runtime
combination.

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
| `generations/0001/md/` | GPUMD trajectories and health report |
| `generations/0001/select/` | FPS selection result |
| `generations/0001/label/selected-labels.xyz` | New ABACUS labels |
| `generations/0001/label/label-provenance.json` | INPUT, UPF/ORB hashes, and backend provenance |
| `generations/0001/dataset/` | Merged training set |
| `generations/0001/retrain/` | Retrained model and PNG convergence plot |
| `generations/0001/evaluate/` | Evaluation result and PNG plots |

A final `budget_exhausted` state means the example used its one-generation
budget. Determine real failures from stage/job states and logs.

## 9. Convert it to a production project

- Replace EMT data with labels from the same ABACUS theoretical level.
- Pin UPF files for every element and ORB files for LCAO.
- Review `ecutwfc`, k points, smearing, magnetism, and SCF convergence.
- Replace the 10–80-step NVE path with validated sampling.
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
