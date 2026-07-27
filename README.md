# NepTrain

NepTrain 是 NEP 模型生命周期的统一命令行工具。它既能独立运行训练、MD、结构
标注和采样，也能把同一套步骤组合成可恢复的主动学习 workflow。

```text
train → md → select → label → merge → retrain → evaluate
```

手动命令和自动 workflow 使用相同的训练、MD、Label Adapter 和执行 target；
workflow 只负责计划、状态推进和验收，不复制科学计算逻辑。Label Adapter 可以
是 VASP、ABACUS，也可以是已经微调并用于替代 DFT 的等变 Teacher 模型。

## 安装

```bash
pip install NepTrain
```

TorchNEP 训练需要安装与机器 CUDA 匹配的 PyTorch：

```bash
pip install torch
pip install 'NepTrain[torchnep]'
```

不提供 `--nep`、需要用 SOAP 做手动采样时安装：

```bash
pip install 'NepTrain[soap]'
```

LAMMPS、VASP 和 ABACUS 由用户或计算平台提供。使用 NEPAdapters LAMMPS plugin
时设置：

```bash
export LAMMPS_PLUGIN_PATH=/path/to/nepadapters/lib
```

NepTrain 只接受 `schema_version: 8`。旧 `train/vasp/gpumd/nep` 命令和旧配置不再
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

### 批量 MD

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

将 `--backend` 改为 `gpumd` 即可使用 GPUMD。无模板时 NepTrain 会生成 NVT/NPT
输入；`--seed` 控制初速度随机种子，`--pressure` 的 GPUMD 单位为 GPa。提供
`run.in` 模板时，模板仍负责 thermostat/barostat 类型、耦合常数、`time_step`
和 dump 间隔；NepTrain 只写入本轮模型、温度、NPT 目标压强、步数和种子，并确保
dump 包含力。GPUMD 和 LAMMPS 的轨迹都会生成同一格式的健康报告，失败任务可保留
稳定段和炸前帧。Spin MD 仍只支持 LAMMPS DynSpin。

### 批量标注

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

所有 shard 成功后才会按原输入顺序发布最终 `labeled.xyz`。失败时保留成功结果，但
不会把部分结果伪装成完整输出。已有结果默认不会被覆盖；确认替换时显式加
`--force`。

VASP 必须同时提供 `--potcar-manifest`，其中逐元素固定相对 `POTCAR` 路径、
SHA256、精确 `TITEL`、势函数族和发行版本。ABACUS 使用
`--resource-manifest` 固定 UPF 和可选 orbital。NepTrain 在本地准备阶段校验
元素覆盖、文件身份和 hash；远端资源不上传，由 `doctor` 在目标机预检，worker
启动 DFT 前再次校验。资源不匹配时不会启动 VASP/ABACUS。
VASP manifest 的路径必须是 `Fe/POTCAR` 或 `Fe_pv/POTCAR` 这一层形式；
NepTrain 会把该路径反向写入 ASE `setups`，保证“校验的 POTCAR”就是实际拼接
进计算的 POTCAR。

微调后的等变模型使用 `model` Adapter。runner 是安装在 labeling target 环境中的
小型命令，接收 `--model`、`--input`、`--output`、`--device` 和
`--precision`，并输出具有 energy、forces、virial 的 extxyz：

```bash
neptrain label candidates.xyz \
  --backend model \
  --model teacher.model \
  --model-name mace-foundation-finetune \
  --runner mace-neptrain-label \
  --device cuda \
  --structures-per-job 64 \
  -o labeled.xyz
```

NepTrain 会记录 Teacher 模型 SHA256 和运行配置，并在发布前校验结构身份、顺序和
标签完整性。Spin 输入还必须由 runner 真实输出 `mforce`，不会自动补零。

### 手动采样

```bash
neptrain select md-300.xyz md-600.xyz \
  --base train.xyz \
  --nep nep.txt \
  --max-selected 64 \
  --min-novelty 0.01 \
  --out selected.xyz \
  --report selected.selection.json
```

手动命令和 workflow 共用按元素集合分组、按来源条件平衡的 FPS 策略。
`--base` 作为已有训练集 warm start；精确重复结构和描述符重复点不会为填满上限而
再次入选。提供 `--nep` 时使用 NEP 描述符，否则使用 SOAP。JSON 报告记录结构
身份版本、描述符来源、入选 ID、novelty 和各分组统计。

### Slurm target

手动命令不要求项目文件；只有复用 Slurm 或远端环境时才需要 `--project` 和
`--target`：

```bash
neptrain label candidates.xyz \
  --backend vasp \
  --project project.yaml \
  --target label \
  --structures-per-job 1 \
  --max-concurrent 20 \
  -o labeled.xyz
```

提供 `--project` 后，未在命令行覆盖的 backend、输入模板、温度、压强、步数和
运行参数会直接读取 schema-v8 项目；命令行只需要写本次确实要改的值。

本地 `process` target 前台执行。Slurm target 提交后立即返回：

```bash
neptrain task status runs/label-...
neptrain task logs runs/label-...
neptrain task wait runs/label-...
neptrain task retry runs/label-...
neptrain task cancel runs/label-...
```

加 `--wait` 可以在提交后等待并自动收集结果。
同一平台的共享文件系统会由最后完成的 array task 自动发布结果；跨平台 target
需要运行 `task status` 或 `task wait` 将远端结果同步回来。

这些命令的边界是：

- `status`：进行一次有超时的 scheduler/SSH 查询，并同步已经原子发布的结果；
  不持续等待。
- `wait`：重复执行 `status`，直到 `complete`、`failed`、`cancelled` 或
  `damaged`。
- `cancel`：请求取消当前 Slurm job；已校验完成的 shard 保留。
- `retry`：创建新 attempt，只提交失败、取消、缺失或损坏的 shard。
- `logs`：同步并列出 scheduler 日志，不改变科学结果。

普通进度和诊断只写 stderr。`--json` 的 stdout 只包含一个 JSON 对象；手动状态
协议为 `neptrain.manual-status.v1`，可稳定读取 `state`、`jobs`、`errors` 和
`next_action`。

## 自动 workflow

创建一个严格的 schema-v8 项目：

```bash
neptrain workflow init \
  --profile slurm \
  --ensemble npt \
  --dft-backend vasp \
  --directory fe-project
cd fe-project
```

补齐 `train.xyz`、`validation.xyz`、`nep.in`、`structures/`、标注输入和环境脚本，
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
neptrain workflow run workflow
```

也可以跳过单独准备，直接执行
`neptrain workflow run project.yaml` 完成创建并启动。

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

`stop` 默认同时停止 workflow Controller，并取消当前排队或运行的计算任务。取消记录
会写入 workflow 历史；以后恢复时会为该阶段创建新的可追踪 attempt。
如果只想暂时停止 Controller、保留当前任务继续计算：

```bash
neptrain workflow stop workflow --keep-jobs
```

独立手动任务使用 `neptrain task cancel` 明确取消。

`workflow run` 同时接受项目 YAML 和已准备的 workflow 目录。第一次启动
`--prepare-only` 的目录优先使用 `workflow run`；`workflow resume` 用于恢复
暂停、失败或中断过的流程。

## schema v8

自动采样只有 `sampling.routes` 一个权威位置。每条 route 显式绑定结构、MD
模板和温度路径。默认压强、递进策略、轨迹帧策略和 FPS 上限无需重复填写：

```yaml
schema_version: 8

training:
  backend: torchnep
  initial_path: ./train.xyz
  config_path: ./nep.in
  device: cuda

md:
  backend: lammps
  inference_backend: auto
  spin: false

sampling:
  routes:
    - id: default
      structures: [./structures]
      template_path: ./lammps.in
      conditions:
        temperature_path: [300, 500, 700]

labeling:
  backend: vasp
  input_path: ./INCAR
  resource_path: /shared/potpaw_PBE
  potcar_manifest_path: ./vasp-resources.json
  kpoint_mode: auto

evaluation:
  validation_path: ./validation.xyz
  inference_backend: auto
  max_rmse:
    energy_rmse: 0.05
    force_rmse: 0.20

workflow:
  id: fe-workflow
  max_model_generations: 12
  seed: 20260721

execution:
  stage_targets:
    training: v100
    sampling: cpu
    labeling: label
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
    label:
      executor: slurm
      partition: compute
      time: 24:00:00
      cpus_per_task: 32
      setup_script: ./env-label.sh
      environment:
        NEPTRAIN_VASP_COMMAND: srun vasp_std
```

`md` 只选择 MD Adapter 和推理后端。结构、MD 模板、温度、压强和递进步数
都由 `sampling.routes` 管理。
轨迹健康检查和 FPS 批量上限统一放在 `sampling`。`timestep`、`tdamp`、`pdamp`、
`spin_alpha` 和 dump 频率直接写在用户的 MD 模板中。

`labeling.kpoint_mode: auto` 优先保留 INCAR 中的 `KSPACING`/`KGAMMA`
或 ABACUS INPUT 中的 `kspacing`；生成的默认输入已给出可直接修改的
`0.2`。只有显式选择 `kspacing` 或 `kpoints` 模式时，NepTrain 才接管
k 点设置。

远程 SSH labeling target 必须设置该目标机自己的绝对
`labeling_resource_path`；本地 `labeling.resource_path` 不会被假定为远端同一路径。
大型资源库不进入 task archive，只有小型内容寻址 manifest 会进入任务身份。

`temperature_path` 是严格有序的温度探路路径，只有前一个温度通过才解锁下一个。
默认全部温度都会继续跑 short、long 和 production。显式设置
`production_temperatures` 后，其余温度只做便宜的 smoke 探路。所有通过健康检查
的 dump 帧都会参与全局 FPS；系统只会去除
训练集已有结构和完全重复结构，不会在描述符计算前按 stride 或数量上限抽帧。
`max_selected` 是一个采样轮最多送去 Label Adapter 的结构数。系统自动把常规
积累下限设为它的一半（默认 100 对应 50）；当前场景 frontier 耗尽或物理失败
抢救时可以提前 flush，不需要用户再维护一组批量阈值。

模型版本之间是硬边界：每轮候选只允许来自该轮激活模型，候选文件和 manifest
都会记录模型哈希。标签诊断超阈值或 MD 发生物理失败时才更新模型；只有新模型完成
evaluate 并写入 lineage，下一轮 MD 才会启动。若当前模型已通过新标签诊断，则
保持该模型继续做长时认证，避免无意义更新反复清零稳定性证据。因此不会把旧模型
轨迹误当成新模型的采样证据。

每个 replica 都会得到 NepTrain 派生的确定性 `{{ seed }}`；用户模板可把它同时
用于 `velocity create` 和 DynSpin thermostat，保证重复运行可复现但 replica
彼此独立。`novelty: auto` 不伪造误差校准：选择阶段接受所有正新颖度候选，
完成条件要求当前池没有剩余覆盖缺口。需要显式阈值时，可同时配置
`selection_threshold` 和 `completion_threshold`。

`workflow.max_model_generations` 是最大模型代数预算。生产温度、最长时长、replica、轨迹诊断和
validation 全部通过后会提前完成；预算耗尽或连续无进展会分别报告
`budget_exhausted` 或 `stalled`，不会误报为 `complete`。

LAMMPS plugin 由 `execution.targets.*.setup_script` 加载，例如在
`env-cpu.sh` 中执行 `module load lammps/nep-release` 或设置
`LAMMPS_PLUGIN_PATH`；NepTrain 不接受 `plugin_path`，也不会重复执行
`plugin load`。

未知字段、拼写错误和旧字段会直接失败。CLI 覆盖值、输入快照、target、作业号和最终
生效配置都会进入运行记录。

外部程序默认从 `PATH` 查找：`nep`、`gpumd`、`lmp`、`vasp_std` 和 `abacus`。
平台命令不同时，在 target 的 `environment` 中设置
`NEPTRAIN_NEP_COMMAND`、`NEPTRAIN_GPUMD_COMMAND`、
`NEPTRAIN_LMP_COMMAND`、`NEPTRAIN_MPIEXEC`、`NEPTRAIN_VASP_COMMAND` 或
`NEPTRAIN_ABACUS_COMMAND`，不再读取用户目录下的隐式配置。自动 MD 的 MPI
rank 数和 VASP/ABACUS 的 CPU 数取当前任务的 `SLURM_CPUS_PER_TASK`；手动命令仍可用
`--mpi-ranks` 或 `--cpus` 显式指定。

如果第一性原理标注位于另一台 Slurm 超算：

```yaml
    remote-label:
      executor: slurm
      host: other-cluster
      work_root: ~/neptrain-runs
      command: /path/to/python -m NepTrain.cli.cli
      partition: compute
      cpus_per_task: 32
      labeling_resource_path: /shared/pseudopotentials/PBE
```

大型赝势库不会跨平台复制；远端路径必须通过 target 的
`labeling_resource_path` 明确给出。target 不能覆盖温度、DFT 精度或 validation 等
科学配置。

## Spin 数据契约

Spin 和磁力直接使用 extxyz：

```text
Properties=species:S:1:pos:R:3:spin:R:3:mforce:R:3
```

- `spin` 是完整磁矩向量，方向和模长都可演化。
- `mforce` 是参考磁力 `-dE/dspin`。
- Spin 结构必须得到 `mforce`，否则标注失败。
- VASP 允许 `ISPIN=1` 或共线 `ISPIN=2`，但两者都只发布普通
  energy/force/virial 标签；不会把共线磁计算伪装成 `spin/mforce` 数据。
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
  --steps 100000 \
  -o spin-trajectory.xyz
```

旧数据若使用 `spins`/`mforces` 等别名，不会在正常训练或合并时静默猜测。
显式迁移并保留原文件：

```bash
neptrain data migrate-spin legacy.xyz canonical.xyz --json
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

状态权威只有一条链：

- `.neptrain/manifest.json`：prepare 时固定的 workflow 实例、输入和计划身份，
  不记录运行态。
- `.neptrain/ledger.json`：已提交科学阶段和代的唯一权威；写入它才算 commit。
- `.neptrain/controller.json`：当前执行 intent、job handle、attempt 和取消历史，
  不能覆盖 ledger 的科学事实。
- 每个 job 的 `task.json` 是不可变输入契约，`execution.json` 是运行观察，
  `result.json` 是待校验结果；只有通过内容、身份和 artifact 校验后才进入 ledger。
- `results/accepted/` 是不可变发布，`results/current` 和根部链接只是可重建投影。

删除可见结果链接后，`workflow resume` 可从 hash 一致的已提交副本修复；删除或
篡改 ledger、唯一的已提交 artifact 或不可变 publication 会得到 `damaged`，
且 resume 在启动任何新任务前 fail closed。不要把删除 `.neptrain` 当作“重置”；
需要重新开始时创建新的 workflow 目录，它会得到新的随机 instance id，不会复用
旧 task/result。

每代目录直接是 `train/`、`md/`、`select/`、`label/`、`diagnose/`、
`dataset/`、`retrain/` 和 `evaluate/`。训练输出、loss 和模型发布到对应阶段
目录，`calculation` 软链指向真实执行目录。训练阶段会用 Matplotlib 发布
`training-convergence.svg` 和 `training-report.json`；配置独立验证集时，
evaluate 阶段还会发布按阈值归一化的 `evaluation-metrics.svg`，以及 Energy、
Force、Virial（spin 模型另含 magnetic force）的 reference/prediction parity 图
`evaluation-parity.svg`。每张图都有对应的 JSON 报告记录数据来源、点数和 RMSE。

开发阶段的确定性工作流 smoke：

```bash
neptrain smoke --profile ordinary
neptrain smoke --profile spin --force
neptrain smoke --profile recovery --force
```
