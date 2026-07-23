# NepTrain

NepTrain 是 NEP 模型生命周期的统一命令行工具。它既能独立运行训练、MD、DFT
标注和采样，也能把同一套步骤组合成可恢复的主动学习 workflow。

```text
train → md → select → dft → merge → retrain → evaluate
```

手动命令和自动 workflow 使用相同的训练、MD、DFT Adapter 和执行 target；workflow
只负责计划、状态推进和验收，不复制科学计算逻辑。

## 安装

```bash
pip install NepTrain
```

TorchNEP 训练需要安装与机器 CUDA 匹配的 PyTorch：

```bash
pip install torch
pip install 'NepTrain[torchnep]'
```

LAMMPS、VASP 和 ABACUS 由用户或计算平台提供。使用 NEPAdapters LAMMPS plugin
时设置：

```bash
export LAMMPS_PLUGIN_PATH=/path/to/nepadapters/lib
```

NepTrain 只接受 `schema_version: 4`。旧 `train/vasp/gpumd/nep` 命令和旧配置不再
兼容，也不会被静默迁移。

## 独立运行一个步骤

### 训练

```bash
neptrain train train.xyz \
  --backend torchnep \
  --config nep.in \
  --device cuda \
  -o nep.txt
```

训练后立即通过 NEPAdapters 检查模型格式和 spin 能力。TorchNEP 的最佳模型统一发布
为 `nep.txt`。

### 批量 LAMMPS

```bash
neptrain md structures/ \
  --backend lammps \
  --model nep.txt \
  --temperature 300 500 700 \
  --pressure 0 \
  --steps 100000 \
  --max-concurrent 12 \
  -o trajectories.xyz
```

输入会按“结构 × 温度”展开成独立任务。使用 Slurm target 时，它们会成为一个带并发
上限的 job array，而不是在登录终端串行运行。

### 批量 VASP 或 ABACUS

```bash
neptrain dft candidates.xyz \
  --backend vasp \
  --input-file INCAR \
  --resources /shared/potpaw_PBE \
  --structures-per-job 1 \
  --max-concurrent 20 \
  -o labeled.xyz
```

所有 shard 成功后才会按原输入顺序发布最终 `labeled.xyz`。失败时保留成功结果，但
不会把部分结果伪装成完整输出。已有结果默认不会被覆盖；确认替换时显式加
`--force`。

### Slurm target

手动命令不要求项目文件；只有复用 Slurm 或远端环境时才需要 `--project` 和
`--target`：

```bash
neptrain dft candidates.xyz \
  --backend vasp \
  --project project.yaml \
  --target dft \
  --structures-per-job 1 \
  --max-concurrent 20 \
  -o labeled.xyz
```

提供 `--project` 后，未在命令行覆盖的 backend、输入模板、温度、压强、步数和
运行参数会直接读取 schema-v4 项目；命令行只需要写本次确实要改的值。

本地 `process` target 前台执行。Slurm target 提交后立即返回：

```bash
neptrain task status runs/dft-...
neptrain task logs runs/dft-...
neptrain task wait runs/dft-...
neptrain task retry runs/dft-...
neptrain task cancel runs/dft-...
```

加 `--wait` 可以在提交后等待并自动收集结果。
同一平台的共享文件系统会由最后完成的 array task 自动发布结果；跨平台 target
需要运行 `task status` 或 `task wait` 将远端结果同步回来。

## 自动 workflow

创建一个严格的 schema-v4 项目：

```bash
neptrain workflow init --profile slurm --directory fe-project
cd fe-project
```

补齐 `train.xyz`、`validation.xyz`、`nep.in`、`structures/`、DFT 输入和环境脚本，
然后检查：

```bash
neptrain doctor --project project.yaml
```

先准备而不提交：

```bash
neptrain workflow run project.yaml --prepare-only
```

确认项目快照后运行：

```bash
neptrain workflow run project.yaml
```

输出目录默认使用 `workflow.id`，也可用 `--output` 覆盖。Controller 默认脱离终端，
不占计算节点，也不依赖 Slurm `afterok`。

日常控制：

```bash
neptrain workflow status workflow
neptrain workflow status workflow --jobs
neptrain workflow resume workflow
neptrain workflow stop workflow
neptrain workflow extend workflow 5
```

`stop` 默认只停止 workflow Controller，不取消已在运行的计算任务；恢复后会接管同一任务。
如果本次流程已经确定作废，同时取消当前排队或运行的计算任务：

```bash
neptrain workflow stop workflow --cancel-jobs
```

取消记录会写入 workflow 历史；以后恢复时会为该阶段创建新的可追踪 attempt。
独立手动任务使用 `neptrain task cancel` 明确取消。

## schema v4

温度、压强和 MD 步数只有 `md` 一个权威位置：

```yaml
schema_version: 4

training:
  backend: torchnep
  initial_path: ./train.xyz
  config_path: ./nep.in
  device: cuda
  torch_backend: auto
  precision: float32
  use_compile: false
  finetune_lr_scale: 0.1
  seed: 20260723

md:
  backend: lammps
  inference_backend: auto
  structures: ./structures
  template_path: ./lammps-nvt.in
  ensemble: nvt
  temperatures: [300, 500, 700]
  pressure: 0.0
  initial_steps: 10000
  timestep: 0.001
  tdamp: 0.1
  pdamp: 1.0
  dump_interval: 100
  spin: false
  mpi_ranks: 4
  lmp: lmp
  mpiexec: mpirun

dft:
  backend: vasp
  n_cpu: 4
  input_path: ./INCAR
  resource_path: /shared/potpaw_PBE
  kpoints_use_gamma: true
  use_k_stype: kspacing
  kspacing: 0.2

evaluation:
  validation_path: ./validation.xyz
  inference_backend: auto
  max_rmse:
    energy_rmse: 0.05
    force_rmse: 0.20

workflow:
  id: fe-workflow
  generations: 3
  seed: 20260721
  initial_candidates: 200
  dft_budget: 20
  minimum_dft_budget: 8
  frame_stride: 2
  min_distance: 0.0
  maturity:
    enabled: true

execution:
  poll_interval: 30
  routes:
    training: v100
    sampling: cpu
    labeling: dft
    analysis: cpu
  targets:
    v100:
      executor: slurm
      partition: 16V100
      qos: flood-1o2gpu
      time: 24:00:00
      gpus_per_node: 1
      setup_script: ./env-training.sh
    cpu:
      executor: slurm
      partition: DSPRHBM
      qos: rush-cpu
      time: 04:00:00
      cpus_per_task: 4
      setup_script: ./env-cpu.sh
    dft:
      executor: slurm
      partition: compute
      time: 24:00:00
      cpus_per_task: 32
      setup_script: ./env-dft.sh
      environment:
        NEPTRAIN_VASP_COMMAND: srun vasp_std
```

未知字段、拼写错误和旧字段会直接失败。CLI 覆盖值、输入快照、target、作业号和最终
生效配置都会进入运行记录。

外部程序默认从 `PATH` 查找：`nep`、`gpumd`、`lmp`、`vasp_std` 和 `abacus`。
平台命令不同时，在 target 的 `environment` 中设置
`NEPTRAIN_NEP_COMMAND`、`NEPTRAIN_GPUMD_COMMAND`、
`NEPTRAIN_VASP_COMMAND` 或 `NEPTRAIN_ABACUS_COMMAND`，不再读取用户目录下的
隐式配置。

如果 DFT 位于另一台 Slurm 超算：

```yaml
    remote-dft:
      executor: slurm
      host: other-cluster
      work_root: ~/neptrain-runs
      command: /path/to/python -m NepTrain.cli.cli
      partition: compute
      cpus_per_task: 32
      overrides:
        dft.resource_path: /shared/pseudopotentials/PBE
```

大型赝势库不会跨平台复制；远端路径必须通过 target `overrides` 明确给出。

## Spin 数据契约

Spin 和磁力直接使用 extxyz：

```text
Properties=species:S:1:pos:R:3:spin:R:3:mforce:R:3
```

- `spin` 是完整磁矩向量，方向和模长都可演化。
- `mforce` 是参考磁力 `-dE/dspin`。
- Spin 结构必须得到 `mforce`，否则标注失败。
- VASP 当前只支持非磁标注。
- ABACUS DeltaSpin 支持全矢量约束并读取最终 magnetization 和 mforce。
- LAMMPS DynSpin dump 根据 `compute property/atom` 定义解析，不写死
  `c_spin[n]` 的意义。

Spin MD 示例：

```bash
neptrain md spin.xyz \
  --backend lammps \
  --model nep.txt \
  --spin \
  --temperature 300 \
  --spin-temperature 500 \
  --spin-alpha 0.01 \
  --steps 100000 \
  -o spin-trajectory.xyz
```

## Workflow 产物

```text
workflow/
├── project.yaml
├── inputs/
├── results/
├── generations/
├── logs/
└── .neptrain/
```

`results/` 只发布最新通过验收的 `nep.txt`、`train.xyz` 和指标。
`generations/` 保存每代科学证据，内部任务、锁、manifest 和 ledger 放在
`.neptrain/`。

开发阶段的确定性工作流 smoke：

```bash
neptrain smoke --profile ordinary
neptrain smoke --profile spin --force
neptrain smoke --profile recovery --force
```
