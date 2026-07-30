<div align="center">
<strong>English</strong> | <a href="README.zh-CN.md">简体中文</a>
</div>

# NepTrain

NepTrain is a command-line tool for the complete lifecycle of neuroevolution
potential (NEP) models. It can run training, molecular dynamics, structure
selection, and labeling as standalone tasks, or compose the same steps into a
resumable active-learning workflow.

```text
train → md → select → label → merge → retrain → evaluate
```

Standalone commands and automated workflows use the same scientific adapters
and execution targets. The workflow layer owns planning, state transitions, and
acceptance; it does not duplicate the scientific calculation logic.

## What NepTrain provides

- GPUMD or TorchNEP training with a canonical `nep.txt` result.
- GPUMD or LAMMPS sampling, including LAMMPS DynSpin for spin MD.
- VASP, ABACUS, MACE, DeepMD/DPA, or TACE labeling.
- Element-set-aware farthest-point sampling with deterministic provenance.
- Local process, local Slurm, and SSH + Slurm execution targets.
- Content-addressed task bundles, atomic publication, hash validation, and
  resumable workflow state.
- PNG training-convergence and evaluation reports generated with Matplotlib.

## Installation

NepTrain requires Python 3.10 or later:

```bash
pip install NepTrain
```

Install only the optional runtime you need:

```bash
# TorchNEP training
pip install torch
pip install 'NepTrain[torchnep]'

# MACE teacher labeling
pip install 'NepTrain[mace]'

# DeepMD / DPA teacher labeling
pip install 'NepTrain[deepmd]'

# TACE teacher labeling (TACE is currently installed from its source repository)
pip install \
  'TACE @ git+https://github.com/xvzemin/tace.git@4b977dcc13ee87d8ba6cceba3ffb7abe43c087c8'

# Optional cuEquivariance acceleration on Ampere-or-newer CUDA 12 GPUs
pip install \
  'TACE[cueq12] @ git+https://github.com/xvzemin/tace.git@4b977dcc13ee87d8ba6cceba3ffb7abe43c087c8'
export TACE_USE_CUE=1

# SOAP descriptors when manual selection has no NEP model
pip install 'NepTrain[soap]'
```

LAMMPS, GPUMD, VASP, ABACUS, and first-principles resource files are supplied
by the user or computing platform.

## First successful run

Use the deterministic toy workflow to check the NepTrain installation without
submitting DFT or Slurm jobs:

```bash
neptrain smoke --profile ordinary
```

For a real project, validate every configured execution target before
submission:

```bash
neptrain doctor --project project.yaml
```

`doctor` reads the project backends, stage targets, setup scripts, and target
environment. It checks the actual GPUMD/TorchNEP, LAMMPS/GPUMD,
VASP/ABACUS, or MACE/DeepMD/TACE runtime required by each target.

## Standalone steps

Train one model:

```bash
neptrain train train.xyz \
  --backend torchnep \
  --config nep.in \
  --device cuda \
  -o nep.txt
```

Run a structure × temperature MD batch:

```bash
neptrain md structures/ \
  --backend lammps \
  --model nep.txt \
  --temperature 300 500 700 \
  --steps 100000 \
  --max-concurrent 12 \
  -o trajectories.xyz
```

Select representative structures:

```bash
neptrain select md-300.xyz md-600.xyz \
  --base train.xyz \
  --nep nep.txt \
  --max-selected 64 \
  --out selected.xyz \
  --report selected.selection.json
```

Label structures with VASP:

```bash
neptrain label candidates.xyz \
  --backend vasp \
  --input-file INCAR \
  --resources /shared/potpaw_PBE \
  --potcar-manifest vasp-resources.json \
  --structures-per-job 1 \
  --max-concurrent 20 \
  -o labeled.xyz
```

Submitted standalone tasks use one control surface:

```bash
neptrain task status runs/label-...
neptrain task logs runs/label-...
neptrain task wait runs/label-...
neptrain task retry runs/label-...
neptrain task cancel runs/label-...
```

The final output is published only after every shard passes identity, order,
and artifact validation. Existing output is not overwritten unless `--force`
is given.

## Automated workflow

Create a schema-v8 project:

```bash
neptrain workflow init \
  --profile slurm \
  --ensemble npt \
  --dft-backend vasp \
  --directory fe-project
cd fe-project
```

After filling in structures, training data, templates, resource manifests, and
execution targets:

```bash
neptrain doctor --project project.yaml
neptrain workflow run project.yaml --prepare-only
neptrain workflow run fe-workflow
```

Inspect and control it with:

```bash
neptrain workflow status fe-workflow --jobs
neptrain workflow resume fe-workflow
neptrain workflow stop fe-workflow
neptrain workflow extend fe-workflow 5
```

NepTrain accepts only `schema_version: 8`. Unknown fields and legacy project
formats fail explicitly instead of being migrated silently.

Optional Feishu progress notifications can be configured directly in
`project.yaml`:

```yaml
notifications:
  feishu:
    webhook: https://open.feishu.cn/open-apis/bot/v2/hook/REPLACE
    secret: REPLACE
    timeout_seconds: 5
```

`neptrain doctor --project project.yaml` sends a real connectivity probe.
Workflow delivery runs on a background thread and is always best effort:
notification failures never change workflow state, exit status, or the
scientific ledger. Progress and terminal messages include both the workflow id
and absolute workflow path so concurrent runs remain distinguishable.

## Labeling backends

| Backend | Runtime | Main boundary |
|---|---|---|
| VASP | User-provided VASP and pinned POTCAR manifest | Ordinary energy/force/virial labels |
| ABACUS | User-provided ABACUS and pinned UPF/ORB manifest | Ordinary or DeltaSpin spin/mforce labels |
| MACE | `NepTrain[mace]` | Ordinary structures; no `mforce` |
| DeepMD / DPA | `NepTrain[deepmd]` | DPA-3 and supported DPA-4 formats; no `mforce` |
| TACE | Official TACE source install | Checkpoints supported by `tace-eval`; spin requires real noncollinear magnetic-force output |

Teacher-model labeling still enters through `neptrain label`. The hidden
`model-worker` command is an internal runner protocol, not a second user-facing
CLI.

## Documentation and tutorials

- [English documentation](https://neptrain.readthedocs.io/en/latest/)
- [中文文档](https://neptrain.readthedocs.io/zh-cn/latest/)
- [VASP + Slurm workflow](examples/workflow-vasp-slurm/README.en.md)
- [ABACUS + Slurm workflow](examples/workflow-abacus-slurm/README.en.md)
- [DeepMD / DPA distillation](examples/distillation-deepmd/README.en.md)
- [MACE distillation](examples/distillation-mace/README.en.md)
- [TACE distillation](examples/distillation-tace/README.en.md)

The website documentation is built from one Chinese source tree with complete
English translation catalogs. Both languages are checked with strict Sphinx
warnings in CI.

## Scientific-data boundaries

- Structure identity uses the versioned `structure-id.v3` contract.
- Spin datasets use canonical `spin` and `mforce` extxyz arrays.
- VASP and ABACUS resources are pinned by relative path and SHA256 manifest.
- Model, dataset, task, result, and publication identities are content
  addressed.
- Workflow ledger entries are scientific commit points; controller state is
  execution intent only.

## Support

- [Issue tracker](https://github.com/aboys-cb/NepTrain/issues)
- [PyPI](https://pypi.org/project/NepTrain/)
