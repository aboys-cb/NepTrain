# 自动迭代

## 创建项目

```bash
neptrain workflow init \
  --profile slurm \
  --ensemble npt \
  --dft-backend vasp \
  --directory fe-project
cd fe-project
neptrain doctor --project project.yaml
```

项目只接受 `schema_version: 8`。自动采样只读取 `sampling.routes`，不再接受
全局 `md.structures`、`md.template_path`、`sampling.conditions` 或
`sampling.progression`，也不在运行时迁移旧配置。

当前配置职责如下：

- `md`：选择 LAMMPS/GPUMD 和推理后端。
- `labeling`：选择 VASP、ABACUS 或微调后的等变 Teacher 模型。
- `sampling.routes`：每条采样路径显式绑定结构、模板和温度路径。
- `sampling.candidate_pool` / `sampling.selection`：可选的高级覆盖项。
- LAMMPS 模板：物理过程、`timestep`、阻尼、spin 积分参数和 dump 频率。
- `execution.targets.*.setup_script`：module、Python 环境和 LAMMPS plugin。

`workflow init` 默认只生成当前选择的 `lammps.in` 和标注后端输入，不再生成四套
LAMMPS 模板及两种第一性原理输入。Spin 流程使用
`--spin --dft-backend abacus`。

最小 route 只需要：

```yaml
sampling:
  routes:
    - id: default
      structures: [./structures]
      template_path: ./lammps.in
      conditions:
        temperature_path: [300, 600, 900]
```

省略项采用一套固定默认值：压强为 `0`，全部温度都是生产温度，FPS 每轮最多
选择 `100` 个结构，炸前帧和坏尾帧分别为 `2` 和 `1`。默认递进策略为：

```yaml
progression:
  steps:
    smoke_passed: 10000
    short_stable: 40000
    long_stable: 160000
    production_ready: 640000
  replicas:
    smoke_passed: 1
    short_stable: 1
    long_stable: 2
    production_ready: 3
```

只有需要改变这些值时才写进项目。一个 workflow 也可以配置多条 route：

```yaml
sampling:
  routes:
    - id: route_a
      structures: [./structures/a.xyz]
      template_path: ./templates/a.in
      conditions:
        temperature_path: [300, 600, 900]
    - id: route_b
      structures: [./structures/b.xyz]
      template_path: ./templates/b.in
      conditions:
        temperature_path: [500, 1000, 1500]
```

一个 workflow 会调度全部 route。同一个结构可以显式出现在多条 route 中；
`route_id` 和内容指纹会隔离场景成熟度与恢复结果。高级 route 可以显式覆盖
`production_temperatures`、`pressure` 和整套 `progression`。

默认情况下，全部 route 使用 `execution.stage_targets.sampling`。需要把某条
route 送到另一台 Slurm 平台时，只写例外映射：

```yaml
execution:
  stage_targets:
    training: train
    sampling: md-default
    labeling: label
    analysis: cpu
  sampling_route_targets:
    route_b: md-remote
```

NepTrain 只向 LAMMPS 模板注入 `temperature`、`pressure`、`steps`、`seed`、
`replica`、`model_file`、`structure_file` 和输出路径等变量，不管理
`plugin_path`，也不判断模板是 FIRE、NVT/NPT、spin MD、MC+MD 还是 chemical
swap。物理过程、阻尼、积分器和 spin 参数都由模板决定。

每条 route 的 `conditions.temperature_path` 是有顺序的温度探路路径。例如
`[300, 500, 700, 900]` 会先验证 300 K，只有通过后才解锁 500 K。
默认全部温度都会跑到最长时长。显式设置 `production_temperatures` 后，未列入
其中的中间温度只做低成本 smoke 探路。

场景通过后，Controller 会同时解锁下一个温度和当前生产温度的下一档时长。
失败场景保留在原位置，采集稳定段和炸前帧，经 FPS、标注和重训后重试，不会
越过失败温度。`progression.replicas` 控制各时长需要的独立 MD 次数。

所有通过健康检查的 dump 帧都会参与全局选择。FPS 先按精确元素集合分组，按
组大小平方根分配初始名额；组内再按 route、温压条件和轨迹窗口做软平衡，并且
只用相同元素集合的训练结构 warm-start。未用完的名额会确定性转给仍有新颖候选
的元素组。当前使用结构级 NEP descriptor，不把它描述成逐原子局域新颖度。

NepTrain 只在 FPS 前去除训练集已有结构和同一 route 内的完全重复结构，不按固定
stride 抽帧，也不使用候选数量上限提前裁剪。
`sampling.selection.max_selected` 是每个采样轮最多送去 Label Adapter 的结构
数。常规标注下限由系统自动取其一半，例如上限 100 时优先积累至少 50 个；若当前
场景 frontier 已经耗尽，或出现物理失败需要抢救，则允许较小批次提前提交。

这里的“一代”严格绑定一个模型哈希。Controller 会枚举该模型下所有已解锁
scenario attempt，并把它们作为独立 process/Slurm 任务一次性提交；全部进入终态后
再合并候选并执行 FPS。不同 route 可通过 `execution.sampling_route_targets`
送到不同平台。所有候选必须由同一个模型产生；候选池 manifest 和每个 extxyz 帧都会记录
`sampling_model_sha256`。诊断超过阈值或 MD 发生物理失败时才更新模型；新模型
完成 evaluate 并写入 lineage 后，下一轮 MD 才能开始。旧模型轨迹不能混入新模型
候选池。

route 指纹由 route id、模板内容哈希、结构输入内容哈希、规范化 conditions 和
progression 共同生成。场景身份还包含具体结构哈希、温度、压强和采样模型哈希。
因此修改模板或 route 条件后不能复用旧结果；完全相同的配置和内容可以确定性恢复。

## 第一性原理标注与 k 点

默认让用户输入文件管理 k 点：

```yaml
labeling:
  backend: vasp
  input_path: ./INCAR
  resource_path: /shared/potpaw_PBE
  potcar_manifest_path: ./vasp-resources.json
  kpoint_mode: auto
  structures_per_job: 1
  max_concurrent: 20
```

`auto` 会优先保留 VASP INCAR 中的 `KSPACING`/`KGAMMA`，或 ABACUS INPUT
中的 `kspacing`。输入中没有这些设置时，适配器才按默认参数生成网格。

`structures_per_job` 决定每个 VASP/ABACUS 调度任务包含多少个 FPS 选中结构；
`max_concurrent` 限制 workflow 同时运行的标注任务数。默认值分别为 `1` 和
`20`。例如 FPS 选中 100 个结构时，默认生成 100 个单结构任务，最多同时运行
20 个。Controller 保留已完成结构，只重试失败或尚未提交的结构，最后按原始
FPS 顺序校验并合并为 `selected-labels.xyz`。

VASP 的 `potcar_manifest_path` 和 ABACUS 的 `resource_manifest_path` 是
必填 provenance。它们固定逐元素资源相对路径与 SHA256；VASP 还固定精确
`TITEL`、family 和 release。prepare 会先验证所有 sampling route 的元素覆盖，
本地 target 同时验证真实文件。远程资源库不打包上传，远程 labeling target
必须给出自己的绝对 `labeling_resource_path`，并用 `doctor` 在提交前逐文件验证。
manifest 中 `Fe/POTCAR` 或 `Fe_pv/POTCAR` 的 setup 目录会直接驱动 ASE 的
POTCAR 选择，校验路径和计算路径是同一个文件。

确实需要 NepTrain 接管时再显式配置：

```yaml
# 按间距设置
labeling:
  kpoint_mode: kspacing
  kspacing: 0.2
  gamma_centered: true

# 或按网格密度参数设置
labeling:
  kpoint_mode: kpoints
  kpoints: [4, 4, 4]
  gamma_centered: true
```

`kspacing` 和 `kpoints` 不能同时配置。手动命令保持同样规则：
`neptrain label --kspacing 0.2 ...` 或 `neptrain label --ka 4,4,4 ...`；
两者互斥。当前输入接口读取 INCAR/ABACUS INPUT 内的 k 点间距，不把同目录下
单独的 VASP `KPOINTS` 或 ABACUS `KPT` 文件作为自动 workflow 输入。

## 等变 Teacher 模型

当微调后的大模型直接替代 DFT 时，workflow stage 不变，只切换 Label Adapter：

```yaml
labeling:
  backend: model
  model_path: ./teacher.model
  model_name: mace-foundation-finetune
  runner: mace-neptrain-label
  device: cuda
  precision: float32
```

`model_path` 会作为内容寻址输入进入 label task；`model_name`、模型 SHA256、
runner、device 和 precision 会写入 `label-provenance.json`；extxyz 帧本身保留
标签来源、后端、模型名称和模型 SHA256。runner 在
`execution.stage_targets.labeling` 指定的环境执行，因此 Teacher 框架的
PyTorch/CUDA 依赖不需要安装到 Controller 环境。

这一路径在调度上完全替代 DFT，但报告会保留 `teacher_model` 来源，避免把蒸馏
标签误称为 DFT 标签。若配置独立 `evaluation.validation_path`，最终验收仍以该
参考集为准。

## 准备和运行

```bash
neptrain workflow run project.yaml --prepare-only
neptrain workflow run fe-workflow
```

第一条命令只创建不可变输入快照，返回的 `next_action` 可以直接复制执行。
不需要人工检查快照时，也可以直接运行
`neptrain workflow run project.yaml`，一次完成创建和启动。

输出目录默认使用 `workflow.id`。Controller 默认脱离终端，按 ledger 推进：

```text
train → explore → select → label → diagnose → merge → retrain → evaluate
```

训练、MD 和 labeling stage 使用与独立命令相同的 Adapter 和 execution target。

`workflow run <目录>` 只允许状态为 `prepared` 的目录第一次启动。
`workflow resume <目录>` 只用于已经启动过的暂停、失败、中断或可修复损坏状态；
对 prepared 目录会明确要求使用 `run`。失败目录再次执行 `run` 也会被拒绝并指向
`resume`，因此“首次启动”和“恢复”不会共享一条含糊路径。

## 状态与恢复

```bash
neptrain workflow status fe-workflow
neptrain workflow status fe-workflow --jobs
neptrain workflow resume fe-workflow
neptrain workflow stop fe-workflow
neptrain workflow extend fe-workflow 5
```

`workflow status --json` 的 stdout 使用
`neptrain.workflow-status.v1`；run/resume、stop 和 extend 分别使用稳定的
`workflow-control.v1`、`workflow-stop.v1` 和 `workflow-extend.v1`。诊断不混入
JSON stdout。`workflow-control.v1` 始终包含 `workflow_id`、`project`、
`manifest` 和 `action`；`action` 明确区分 `prepare`、`start`、`resume`、
`repair` 与 `noop`，不再另写一份重复的 `started` 布尔状态。

Controller 不依赖 Slurm `afterok`。失败后只重跑 ledger 中未完成的阶段；已完成
workflow 使用 `resume` 是安全 no-op。`workflow run` 接受项目 YAML 或 workflow
目录：对尚未启动的准备目录返回 `action: start`；`workflow resume` 用于暂停、
失败或中断后的恢复。

状态语义如下：

- `prepared`：只有不可变输入快照，下一步是 `workflow run`。
- `running`：Controller lock 存在且当前任务可观察。
- `degraded`：临时 SSH/scheduler 查询失败，Controller 仍保留原 handle 并重试。
- `paused`：Controller 已停止或 PID/lock 不在，但远端工作和 current intent 保留。
- `failed` / `rejected`：执行失败或科学验收失败，可按记录创建新 attempt 恢复。
- `damaged`：已提交 artifact、ledger 或 publication 不满足 hash/身份契约。
  resume 只修复 hash 一致的冗余副本和可见投影；无法证明来源时在提交新任务前
  fail closed。
- `budget_exhausted`：模型代数预算耗尽，先 `workflow extend`，不是成功。
- `stalled`：相同模型与策略继续运行不会增加证据；必须修改科学策略并新建
  workflow，不能原样 retry。
- `complete`：完整 stage 链和（若配置）独立 validation 已验收。

`workflow.max_model_generations` 是最大模型代数预算，不是成功条件。配置 `evaluation`
时，所有生产温度、最长时长、replica、轨迹标签诊断和独立 validation 同时通过
后才会提前结束。预算用尽但仍未收敛时状态为 `budget_exhausted`，连续两轮没有
新覆盖或模型改进时状态为 `stalled`，都不会伪装成 `complete`。

`evaluation` 可以整块省略。此时 workflow 仍可采样、标注、重训和记录场景证据，
但 `validation_accepted` 保持为空，流程不会伪造 validation passed，也不会把
该模型标成已完成独立验证。

默认停止 Controller，并取消当前 process 或 Slurm 作业：

```bash
neptrain workflow stop fe-workflow
```

如果只想暂时退出 Controller，并让当前计算任务继续运行：

```bash
neptrain workflow stop fe-workflow --keep-jobs
```

取消动作会记录到 workflow 历史；后续恢复会创建新的 stage attempt。
对 task group，已完成且通过校验的 shard 保留原 task id；只有未完成、取消或
缺失 shard 生成新 attempt。`--keep-jobs` 仅停止 Controller，下一次 resume
先重新观察原 job handle，不重复提交。作业已离开 `squeue` 时还会查询 accounting；
超过有界 grace 仍不存在才按 LOST 处理。

每代目录直接对应用户关心的阶段：

```text
generations/0001/
├── train/
├── md/
├── select/
├── label/
├── diagnose/
├── dataset/
├── retrain/
└── evaluate/
```

训练模型、loss 和 stdout/stderr 等关键产物会发布到对应阶段目录；
`calculation` 软链指向真实执行目录，便于直接排查。

内部 job 也只保留一层输入和一层输出。job 名称已经包含代数、阶段、route、
attempt 和任务指纹，因此输出目录不再重复这些信息：

```text
.neptrain/jobs/g0001-md-default-.../
├── input/
├── output/
│   ├── log.lammps
│   ├── trajectory.xyz
│   ├── candidates.xyz
│   └── md-attempts.json
├── task.json
├── execution.json
└── result.json
```

训练的 `nep.txt`、`loss.out`、checkpoint 和训练日志同样直接位于该 job 的
`output/`。默认的单结构 VASP/ABACUS job 也把输入、原生输出、后端日志和
`selected-labels.xyz` 直接放在 `output/`；只有发生真实重试时才出现
`retry-0002/`。每代 `label/` 直接发布 `000001-Ce8Fe16` 等软链及最终合并文件，
不再增加 `teacher/`、`attempt-0001/` 或 `calculations/` 层。`input/` 也只携带当前阶段真正消费的文件：训练任务不再
复制 MD route，MD 和标注任务不再复制初始训练集及 validation 数据。

远端 task 使用内容寻址协议：本地先在临时目录完整生成 `task.json` 和全部输入，
校验 manifest 后打包；远端在独占 lock 下解包、复核 hash，再用一次 rename
发布，worker 不会看到半上传 bundle。结果先写到临时 `output`，完成后原子切换并
发布绑定 task id/spec hash 的 `result.json`；收集端只接受属于当前 workflow
instance 的完整结果。旧 workflow 目录删除后重新创建会得到新的随机
`instance_id`，即使配置相同也不能复用旧远端结果。

## 状态权威与目录恢复

- `.neptrain/manifest.json` 只固定 prepare 输入、计划和 workflow instance。
- `.neptrain/ledger.json` 是已提交科学阶段的唯一权威和 commit point。
- `.neptrain/controller.json` 只保存 attempt、current execution、job handle、
  取消与恢复 intent。
- job 内 `task.json` 是不可变输入；`execution.json` 是运行观察；
  `result.json` 是通过收集校验前的候选结果。
- `results/accepted/` 保存不可变发布；`results/current`、根部结果链接和
  `generations/*/calculation` 是相对链接投影，可由权威记录重建。

因此，删除可见链接不会丢失科学状态，`resume` 可修复。删除一个有 hash 冗余副本
的 stage artifact 时，resume 只会复制完全相同的已验证内容；找不到一致副本、
ledger 缺失、publication 唯一副本损坏或 artifact 路径逃出 workflow 时，状态保持
`damaged`，不会自动回退成 prepared 或启动下游阶段。要彻底重跑请创建新目录，
不要删除 `.neptrain` 或手工编辑 JSON 来“重置”。
