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

These examples validate the software path; they do not produce a potential
ready for publication or production simulation. The VASP and ABACUS seed data
come from ASE EMT and must be replaced with first-principles data at the same
theoretical level as the production labels.
