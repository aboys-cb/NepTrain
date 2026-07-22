# NepTrain

NepTrain 用于组织 NEP 主动学习循环。训练和分子动力学是两个独立选择：

| 工作阶段 | 可选 Adapter |
| --- | --- |
| 训练 | GPUMD NEP、TorchNEP（第一阶段单卡） |
| MD | GPUMD、LAMMPS |
| Python 推理与描述符 | NEPAdapters CPU/CUDA |
| LAMMPS pair | NEPAdapters `nep/cpu`、`nep/gpu/kk` |
| DFT 标注 | VASP、ABACUS、Toy Teacher（开发） |

LAMMPS 只是 MD frontend；模型识别、能力检查和计算实现统一由 NEPAdapters 提供。

当前生产保证先收敛在**非磁、固定几何的单点标注**：VASP 和 ABACUS 接受普通
结构并返回 energy、force、virial。spin/mforce 数据结构、Toy Teacher 与 DynSpin
工具仍保留用于后续打磨，但生产 VASP/ABACUS Adapter 会明确拒绝 spin 结构，避免
把尚未真实验收的磁性 DFT 路径误当成可用功能。

## 安装

```bash
pip install NepTrain
```

新的 `run/status/resume` campaign 不依赖 `dpdispatcher`。只有计划在下个版本删除的
旧 `train` 命令需要 `pip install 'NepTrain[legacy]'`；旧 Bohrium 入口使用
`NepTrain[bohrium]`。

TorchNEP 训练需要先安装与机器 CUDA 匹配的 PyTorch，再安装额外依赖：

```bash
pip install torch
pip install 'NepTrain[torchnep]'
```

LAMMPS 由用户安装。使用 runtime plugin 时设置：

```bash
export LAMMPS_PLUGIN_PATH=/path/to/nepadapters/lib
```

## 新项目

```bash
NepTrain init slurm
NepTrain doctor --project job.yaml --training-backend torchnep
NepTrain run job.yaml --output my-campaign
```

`job.yaml` 使用 schema v3。常改参数集中在 `training`、`md`、`dft` 和
`execution`：

```yaml
schema_version: 2
current_job: training

training:
  backend: torchnep        # gpumd 或 torchnep
  initial_path: ./train.xyz
  device: cuda
  config_path: ./nep.in
  finetune_lr_scale: 0.1   # 只作用于增量 retrain；初始训练不变

md:
  backend: lammps          # gpumd 或 lammps
  inference_backend: auto  # auto、cpu 或 cuda
  duration_ps_every_generation: [10, 100, 500]
  temperatures: [300, 500, 700]
  structures: ./structure
  ensemble: npt
  pressure: 0.0
  timestep: 0.001
  mpi_ranks: 4
  # LAMMPS 非零退出时的安全回收窗口；通常无需修改。
  pre_failure_frames: 2
  bad_tail_frames: 1
  health:
    min_distance_ratio: 0.5
    min_volume_ratio: 0.5
    max_volume_ratio: 2.0
    max_force: 100.0
    max_mforce: 100.0
    max_spin_magnitude: 20.0
```

TorchNEP 的 `nep_best.txt` 会复制为循环统一使用的 `nep.txt`。显式选择 CPU/CUDA 时不做静默回退；`auto` 只有在 CUDA runtime 和模型能力都通过时才选择 CUDA。
训练结束后会立即用 NEPAdapters 加载最佳模型；模型格式或 spin descriptor 与运行时不兼容时当场失败，不会等到 MD 阶段才暴露。

预留的可变模长 spin 开发流程使用与 NEPAdapters 一致的 TorchNEP descriptor：

```text
spin_mode 1
spin_descriptor spin_nep_lite
spin_compress 2
spin_basis_size 2 2
spin_l_max 2 0 0
```

TorchNEP 的实验性 `vector_field` descriptor 不是现有 NEPAdapters/LAMMPS runtime 的同一模型协议，不能只改 `spin_mode` header 强行接入。

旧配置显式迁移，不在运行期永久维护两套字段：

```bash
NepTrain migrate old-job.yaml -o job.v3.yaml
```

## 单独运行 MD

普通 LAMMPS NPT：

```bash
NepTrain md structure.xyz \
  --backend lammps \
  --nep nep.txt \
  --ensemble npt \
  --temperature 300 \
  --pressure 0 \
  --steps 100000 \
  --inference-backend auto
```

DynSpin GLSD：

```bash
NepTrain md spin.xyz \
  --backend lammps \
  --nep nep.txt \
  --spin \
  --temperature 300 \
  --spin-temperature 500 \
  --spin-alpha 0.01 \
  --steps 100000 \
  --mpi-ranks 4
```

默认提供 `dynspin/glsd/nvt` 和 `dynspin/glsd/npt` 模板，磁矩方向与模长都会演化。高级用法可通过 `--template custom.in` 提供完整 LAMMPS 输入；NepTrain 只替换模板中实际出现的 `{{ variable }}`。

## 预留的 spin 数据契约

训练和 DFT 标注使用 extxyz：

```text
Properties=species:S:1:pos:R:3:spin:R:3:mforce:R:3
```

- `spin` 是完整物理磁矩向量，模长为 `|spin|`。
- `mforce` 是参考磁力 `-dE/dspin`。
- 当前只有 Toy Teacher 使用这套完整标签契约；生产 VASP/ABACUS 暂不接受 spin。
- 后续磁性 DFT Adapter 启用后，结构包含 `spin` 时必须同时得到 `mforce`，否则失败。
- MD 轨迹中的模型磁力也统一写为 `mforce`，进入 DFT 标注后必须由参考计算结果替换。

默认 DynSpin dump 的含义由 compute 顺序确定：

```lammps
compute spin all property/atom sp spx spy spz fmx fmy fmz fx fy fz
dump dpgen_dump all custom 100 traj.dump id type x y z \
  c_spin[1] c_spin[2] c_spin[3] c_spin[4] \
  c_spin[5] c_spin[6] c_spin[7]
```

因此 `c_spin[1]` 是模长，`c_spin[2:4]` 是方向，`c_spin[5:7]` 是 mforce。解析器会读取 `compute property/atom` 的定义建立映射，而不是把 `c_spin[n]` 的意义写死。

多 MPI rank 不做静态限制。正式任务前可在用户实际 LAMMPS、plugin、模型和 rank 数上运行真实 smoke：

```bash
NepTrain doctor \
  --md-backend lammps \
  --model nep.txt \
  --structure spin.xyz \
  --lmp /path/to/lmp \
  --plugin-path /path/to/nepadapters/lib \
  --mpi-ranks 4
```

## 运行递进 campaign

正式迭代由固定状态机控制：

```text
train → explore → select → label → diagnose → merge → retrain → evaluate
```

每个阶段都写入带 SHA256 的 `.neptrain/ledger.json`。阶段只能按顺序执行，已完成产物被修改、plan 发生漂移、或两个作业试图重复同一阶段时都会停止。

`diagnose` 使用加入新标签前的模型计算 acquisition error，只用于判断这批采样是否确实触及模型薄弱区，不决定该代是否通过。`evaluate` 必须使用合并新标签并重新训练后的模型，以及不进入训练集的固定 validation 数据；只有这个 post-retrain gate 能决定该代是否接受。

Controller 是唯一负责推进状态机的进程；训练、MD、DFT 和分析只是一次只跑一个
stage 的执行任务。Controller 默认脱离终端运行，只做轮询、账本提交和小文件传输，
不占计算节点，也不依赖 Slurm `afterok`。每一类 stage 可以指向本机进程、本机 Slurm，
或另一台可 SSH 登录的 Slurm 集群。

单集群常用配置如下。Controller 放在该集群登录节点运行，训练用单张 V100，MD 和
分析用 4 核 CPU，DFT 可以独立选择资源：

```yaml
campaign:
  id: ordinary-v1
  generations: 3
  seed: 20260721
  initial_candidates: 200
  dft_budget: 20
  minimum_dft_budget: 8
  initial_steps: 10000
  temperatures: [300, 500, 700]
  frame_stride: 2
  maturity:
    # 不写 levels 时自动使用 initial_steps × [1, 4, 16, 64]
    levels:
      smoke_passed: 10000
      short_stable: 40000
      long_stable: 160000
      production_ready: 640000

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
      gpus_per_node: 1
      setup_script: ./env-v100.sh
    cpu:
      executor: slurm
      partition: DSPRHBM
      qos: rush-cpu
      cpus_per_task: 4
      setup_script: ./env-cpu.sh
    dft:
      executor: slurm
      partition: 16V100
      qos: flood-1o2gpu
      cpus_per_task: 4
      gpus_per_node: 1
      setup_script: ./env-dft.sh

dft:
  software: vasp            # vasp、abacus 或 toy
  cpu_core: 1               # DFT 启动的 MPI rank 数
  incar_path: ./INCAR       # ABACUS 时使用 INPUT
  resource_path: ./dft-resources
  kpoints_use_gamma: true
  use_k_stype: kspacing
  kspacing: 0.2

evaluation:
  validation_path: ./validation.xyz
  inference_backend: auto
  max_rmse:
    energy_rmse: 0.05
    force_rmse: 0.20
```

如果 DFT 在另一台超算运行，只改目标和路由，不改科学流程。远端必须能用 SSH alias
登录，并在 `command` 指定的环境中安装相同版本的 NepTrain：

```yaml
execution:
  routes:
    training: sai-v100
    sampling: sai-cpu
    labeling: other-dft
    analysis: local
  targets:
    local:
      executor: process
    sai-v100:
      executor: slurm
      host: sai
      work_root: ~/hpc-work/neptrain-tasks
      command: /path/to/python -m NepTrain.cli.cli
      partition: 16V100
      qos: flood-1o2gpu
      gpus_per_node: 1
      setup_script: /remote/path/env-v100.sh
    sai-cpu:
      executor: slurm
      host: sai
      work_root: ~/hpc-work/neptrain-tasks
      command: /path/to/python -m NepTrain.cli.cli
      partition: DSPRHBM
      qos: rush-cpu
      cpus_per_task: 4
      setup_script: /remote/path/env-cpu.sh
    other-dft:
      executor: slurm
      host: other-cluster
      work_root: ~/neptrain-tasks
      command: NepTrain
      partition: compute
      cpus_per_task: 32
```

没有排队系统时使用 `executor: process`；加上 `host` 和 `work_root` 就是在远端后台
进程中运行。`overrides` 可替换目标机上的安装路径，例如大型赝势库或 LAMMPS plugin，
避免把它们随每个 task 重复传输：

```yaml
    other-dft:
      executor: process
      host: other-cluster
      work_root: ~/neptrain-tasks
      overrides:
        dft.resource_path: /shared/pseudopotentials/PBE
```

VASP 使用 ASE 自带的输入与结果解析能力；`resource_path` 直接指向含
`<元素或 setup>/POTCAR` 的目录。当前 Sai 的 PBE 64 资源部署在
`~/hpc-work/potpaw-pbe64-64/install`。VASP Adapter 要求 `IBRION=-1`、
`NSW=0`、`ISPIN=1`，并为每次重试建立新的 attempt 目录，拒绝把弛豫或磁性模板
当成单点标签。ABACUS Adapter 原生生成
`INPUT/STRU/KPT`、执行 `abacus` 并解析 `OUT.*/running_scf.log`，不需要额外
安装 ASE 插件。ABACUS 的 `resource_path` 放 `.UPF`；`basis_type lcao` 时还
必须放 `.orb`，文件扩展名大小写均可。Sai 的 `env-dft.sh` 加载实际 module，例如
`module load abacus/LTSv3.10.1-sm70-auto` 或
`module load vasp/6.6.0-nvhpc25.7-ompi5.0.10`。module 和赝势是两个独立
依赖，缺任意一项都会在 label 阶段明确失败。

K 点选择不会再靠参数覆盖顺序碰运气：`use_k_stype: kspacing` 使用配置中的
`kspacing`，或在配置未给值时读取 `INCAR/INPUT` 模板；`use_k_stype: kpoints`
则明确生成 KPOINTS/KPT 并忽略模板中的 KSPACING。单独运行 `NepTrain dft` 时，
未给 `--kspacing` 的 `auto` 模式会优先尊重模板 KSPACING，否则使用 `--ka`。

`evaluation.validation_path`（或 `training.test_path`）是 campaign 必填语义。当前
非磁生产流程的 `evaluation.max_rmse` 至少要给出 energy 和 force；空阈值会拒绝
启动，避免“只要数值有限就自动通过”。validation 与合并后的训练集存在重复结构
时也会直接失败，避免用训练误差冒充验收结果。

场景按“初始结构 × 温度 × 压强”建立稳定 ID，MD 步数作为该次证据强度。每个场景只能依次经过 `untested → smoke_passed → short_stable → long_stable → production_ready`，不能跳级。调度时优先运行成熟度最低的场景，所以新增温度会先跑便宜的 smoke，不会直接继承旧温度的长时间等级；只有 MD 完整结束且该代 post-retrain validation 通过，`scenario-maturity.json` 才会晋级。失败证据会保留，但不会晋级。

LAMMPS 非零退出但 dump 中仍有完整帧时，NepTrain 不会丢掉整条轨迹。它先按物理信号寻找第一个异常帧：原子间距相对共价半径过短、cell 体积相对初始结构突变、原子力或 mforce 过大、spin 模长过大以及非有限数都会触发隔离。异常点之前默认两帧标为 `pre_failure`，异常点及其后全部标为 `bad_tail`；如果进程失败但没有检测到物理异常，则保守隔离最后一帧。单项规则可在 `md.health` 中设为 `null` 关闭。

`bad_tail` 只留在每个 MD run 的原始 `trajectory.xyz` 中，不进入候选池；`pre_failure` 会优先进入候选池且不被普通 `frame_stride` 抽稀。普通稳定帧会在整条轨迹的时间范围内均匀铺开后再进入 FPS，避免长 MD 只采到开头。每个 run 的 `trajectory-health.json` 记录第一个异常 timestep、原因、实测值和窗口范围；每代的 `md-attempts.json` 汇总退出原因、最后 timestep 和各窗口帧数。即使 LAMMPS 返回 0，只要轨迹健康检查失败，该场景也不会晋级。若没有任何安全帧，explore 仍会失败关闭。

健康报告还会列出 `available_signals` 和 `unavailable_thresholds`。当前默认 spin dump 包含 spin 与 mforce，但不包含原子力，因此 `max_force` 只有在用户模板把 `fx fy fz`（或 compute 对应列）写入 dump 时才实际执行；系统不会把“没有这个信号”误报为“力已通过检查”。

`env-cpu.sh` 负责站点环境，例如：

```bash
module load lammps/nep-release
```

如果环境需要解压到节点本地盘，`setup_script` 应按环境包版本复用同一个缓存目录，或在作业退出时清理；不要为每个 generation 永久保留一份完整 PyTorch 环境，否则长 campaign 会逐步耗尽节点本地空间。

第一次运行可以先只生成自包含项目目录，不提交作业：

```bash
NepTrain run job.yaml \
  --output ordinary-v1 \
  --prepare-only
```

确认 `ordinary-v1/project.yaml` 和 `inputs/` 后提交：

```bash
NepTrain run ordinary-v1
```

也可以省略 `--prepare-only`，创建后立即启动后台 Controller。命令会很快返回，
不会堵住当前终端；需要在容器或服务管理器中观察前台日志时使用 `--foreground`。
Controller 应运行在允许长驻轻量进程的登录节点，不能放进另一个 batch 作业里嵌套
提交。

日常只需要下面三个命令：

```bash
NepTrain status ordinary-v1
NepTrain resume ordinary-v1
NepTrain stop ordinary-v1
```

`status` 默认只显示科学进度和当前执行状态；需要完整 stage 表时使用
`NepTrain status ordinary-v1 --jobs`。`stop` 只停 Controller，不取消已经在本机、
Slurm 或远端运行的 task；随后 `resume` 会接管同一个 task，不重复提交。执行失败时
`resume` 创建带新 attempt ID 的重试；validation gate 拒绝时则从 retrain 恢复。

项目目录按用户视角组织：

```text
ordinary-v1/
├── project.yaml       # 可读的配置快照，路径指向 inputs/
├── inputs/            # 结构、nep.in、DFT 输入和环境脚本快照
├── results/           # 最新通过验收的 nep.txt、train.xyz 和指标
├── generations/       # 0001、0002……每代的采样/标注/训练/评价证据
├── logs/              # Controller 和执行后端日志
└── .neptrain/         # manifest、ledger、plans、tasks、locks 等内部状态
```

赝势目录和 LAMMPS 插件等大型外部依赖不会复制进项目，但路径和哈希仍进入
manifest；其余小型输入会复制到 `inputs/`，因此原始配置文件移动后不影响已经
准备好的 campaign。

已有 campaign 全部通过后可以原地追加代数，不会重跑或改写历史 generation：

```bash
NepTrain extend ordinary-v1 4
NepTrain run ordinary-v1
```

追加操作只接受更大的总代数，并在 manifest 中记录扩展历史；未完成或被验收门槛拒绝的 campaign 不允许绕过 gate 继续追加。

Controller 只在前一 stage 的结果通过哈希、任务身份和账本校验后创建下一 stage，
因此不需要调度器依赖语法。第一代先 bootstrap；每代依次执行采样、选择、标注、
诊断、合并、retrain 和 validation。TorchNEP retrain 从上一 checkpoint 载入权重，
重新开始优化器和 epoch 计数，并默认使用 `nep.in` 学习率的 `0.1` 倍。下一代直接
复用上一代验收模型。重复启动 Controller 会通过确定性 task ID 和后端查询接管已有
执行，不会重复提交。

恢复只重跑账本确认尚未完成的阶段；被 gate 拒绝时从 `retrain -> evaluate` 开始。
原失败模型、指标和作业号保存在 ledger 的 `recovery_attempts` 中，不会改写成
一次虚假的成功记录。

`.neptrain/manifest.json` 会固定配置、初始训练集、`nep.in`、结构输入、环境脚本、
成熟度策略和计划；`.neptrain/controller.json` 保存当前 target、task ID、执行句柄和
append-only 历史。依赖文件漂移时会拒绝启动。每代的
`scenario-plan.json` 记录实际调度的场景和步数，累计的
`scenario-maturity.json` 保存在同一份 ledger 产物链中，不需要第二套工作流。

内部的 `controller` 和 `stage-worker` 命令只供生成的任务使用，不是用户接口。排障时
优先查看 `NepTrain status --jobs`、`logs/controller.log`、当前 task 的
`execution.json` 和 Slurm 输出。

`generation-1.json` 是 campaign 自动生成的不可变计划，例如：

```json
{
  "generation": 1,
  "seed": 20260721,
  "candidate_count": 200,
  "dft_budget": 20,
  "steps": 10000,
  "temperatures": [300.0],
  "pressure": 0.0,
  "min_distance": 0.0,
  "frame_stride": 2
}
```

真实 Adapter 会用 TorchNEP/GPUMD 训练、LAMMPS/GPUMD 探索、NEPAdapters 描述符做分层 FPS，并根据 `dft.software` 调用 VASP、ABACUS 或 Toy Teacher。MD source 数按候选预算裁剪，已在训练集中的结构不会送入标注。VASP/ABACUS 缺少收敛结果时失败，并且当前会在启动 DFT 前拒绝 spin 结构。

## 开发闭环 smoke

Toy Teacher 只用于打磨工作流，不用于生产标注。它提供解析的 energy、force、virial；spin profile 还提供可变模长 spin-lattice 能量及解析 mforce。每次 smoke 都会用有限差分检查这些导数。

无需 VASP/ABACUS 的快速 contract smoke：

```bash
NepTrain smoke --profile ordinary
NepTrain smoke --profile spin --force
NepTrain smoke --profile recovery --force
```

三个 profile 分别检查普通标签流、`spin:R:3`/`mforce:R:3` 流转，以及中断后分批追加是否得到相同产物。默认输出位于 `outputs/smoke/`，包括完整 teacher truth、实际选中标签和 `smoke-report.json`。teacher truth 只用于反事实评估，不会合并进训练数据。

要检查真正的代际控制而不消耗训练或 DFT 资源，运行三代 Toy campaign。第三代会在保持步数不变时打开第二个温度层：

```bash
NepTrain smoke --profile spin --iterations 3 --force
```

它执行固定的 `train → explore → select → label → diagnose → merge → retrain → evaluate` 顺序，并验证：

- MD 步数增长、候选池增长而 DFT budget 递减；
- 候选先时间抽稀，再按温度/压力分层做确定性 FPS；
- 每个 stage 的输入计划和产物 SHA-256 写入 `campaign-ledger.json`；
- 重启复用已校验产物，任何完成产物漂移都会失败；
- 固定 validation pool 的 coverage 超过容差则拒绝该代并停止下一代；
- 新标签加入前的诊断指标不会误当成 post-retrain acceptance；
- spin 标签在每代 merge 后仍满足 `spin:R:3`/`mforce:R:3`。

这个 campaign 使用本地 Toy coverage surrogate，只验证流程控制和采样策略。真实 TorchNEP/LAMMPS Adapter 仍通过下面的 `--workflow-config` smoke 单独验证。

要检查真实训练和 MD Adapter，传入现有任务配置：

```bash
NepTrain smoke \
  --profile ordinary \
  --workflow-config job.yaml \
  --training-steps 2 \
  --md-steps 2 \
  --force
```

该模式复制配置到 smoke 输出目录，把 DFT Adapter 临时替换成 Toy Teacher，并压缩训练和 MD 步数；TorchNEP/GPUMD 训练、LAMMPS/GPUMD MD、选择、再训练仍使用生产实现。它需要用户配置中的真实后端和运行环境可用，不会用假的 trainer 或 MD 代替失败的依赖。

也可以单独验证 Toy Teacher 标签 seam：

```bash
NepTrain dft candidates.xyz --toy --teacher-profile spin -o labeled.xyz
```

## 运行产物

每一代位于 `cache/Generation-N/`。任务成功后才更新 `restart.yaml`；LAMMPS 非零退出且没有可回收安全帧、缺少最佳模型、缺少轨迹、DFT 不收敛或把 spin 结构交给当前非磁生产 Adapter 都会直接失败。
