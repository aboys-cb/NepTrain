<div align="center">
<strong>English</strong> | <a href="README.md">简体中文</a>
</div>

# NepTrain examples

Choose a route by labeling source. Do not infer which script to run from the
source-tree layout.

| Goal | Example | Label source | External programs |
|---|---|---|---|
| Active learning with VASP | [`workflow-vasp-slurm`](workflow-vasp-slurm/README.en.md) | VASP single-point calculations | TorchNEP, GPUMD, VASP, Slurm |
| Active learning with ABACUS | [`workflow-abacus-slurm`](workflow-abacus-slurm/README.en.md) | ABACUS single-point calculations | TorchNEP, GPUMD, ABACUS, Slurm |
| Distillation from DPA-3/DPA-4 | [`distillation-deepmd`](distillation-deepmd/README.en.md) | DeepMD/DPA teacher | TorchNEP, DeePMD-kit, optionally GPUMD |
| Distillation from MACE | [`distillation-mace`](distillation-mace/README.en.md) | MACE teacher | TorchNEP, MACE, optionally GPUMD |
| Distillation from TACE | [`distillation-tace`](distillation-tace/README.en.md) | TACE teacher | TorchNEP, TACE, optionally GPUMD |

Recommended learning order:

1. Run the standalone labeling command and confirm that the backend produces
   energy, forces, and virial.
2. Run the student smoke training and confirm that the training backend reads
   the labels.
3. Run one workflow generation with the supplied `project.yaml`.
4. Replace the tutorial data, short MD run, and smoke `nep.in` with production
   settings.

These examples teach the software path; they do not supply production-ready potentials.

## Coverage and limits

| Configuration | System and sampling | What it teaches |
|---|---|---|
| VASP `project.yaml` | Periodic Al, GPUMD NPT | Slurm targets, pinned POTCAR, same-backend seed labels |
| VASP `project-lammps.yaml` | Periodic Al, LAMMPS NPT | MD backend replacement; pressure in bar |
| ABACUS `project.yaml` | Periodic Al, GPUMD NPT | Plane waves, pinned UPF, backend replacement |
| MACE / TACE | Periodic Al, GPUMD NPT | Teacher invocation, hashes, and provenance |
| DeepMD / DPA | Single water molecule in vacuum, GPUMD NVT | Multiple elements, model head, fixed-volume molecular labels |
| [DeltaSpin](workflow-vasp-deltaspin/README.en.md) | Periodic Fe, LAMMPS spin NPT, two routes | Vector moments, mforce, and independent route coverage |

Every YAML targets the new workflow; test data are optional. DFT seeds are unlabeled until
processed with the same backend and inputs used later. No EMT/DFT mixing is part of the tutorial.
GPUMD pressure is in GPa; LAMMPS metal pressure is in bar, with no automatic conversion.
The isolated molecule remains NVT. Short trajectories verify interfaces, not equilibration.
Temperature, coupling parameters, timesteps, and acceptance thresholds need system-specific validation.

These cases do not establish scientific acceptance for alloy defects, phase transitions, or large
magnetic training campaigns. Local tests cover YAML, generators, template rendering, and stage contracts;
real backend runs require configured software and resources as described in each tutorial.

## Inspecting results

Start with `neptrain workflow status <workflow-directory>` for the current stage,
the latest three new-label evaluations, convergence requirements, and plot paths.
Use `--details` for all evaluations and optional test diagnostics, or `--jobs`
for grouped tasks. Job progress uses the controller's cached observations; check their timestamps.

Example result paths are relative to the workflow directory, named by `workflow.id` under the project directory by default.
Under `generations/<number>/`, inspect `train/` for loss and parity plots, `select/selection-pca.png` for the
candidate and selected distributions, and `evaluate/acquisition-parity.png` for
predictions on new labels before training on them. Native TorchNEP parity uses
the final epoch, which may differ from the activated best model. With no valid
plot data, only a report explaining the reason is written. JSON is for scripts
and provenance; status and PNG are the usual human-facing views.
