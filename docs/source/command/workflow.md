# 自动迭代

第一次运行建议先从标注后端对应的完整教程开始：

- [VASP + Slurm](https://github.com/aboys-cb/NepTrain/tree/master/examples/workflow-vasp-slurm)
- [ABACUS + Slurm](https://github.com/aboys-cb/NepTrain/tree/master/examples/workflow-abacus-slurm)
- [DeepMD / DPA 蒸馏](https://github.com/aboys-cb/NepTrain/tree/master/examples/distillation-deepmd)
- [MACE 蒸馏](https://github.com/aboys-cb/NepTrain/tree/master/examples/distillation-mace)
- [TACE 蒸馏](https://github.com/aboys-cb/NepTrain/tree/master/examples/distillation-tace)

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
  内置模板使用 `{{ dump_interval }}`：短轨迹约保留 100 帧，长轨迹最多每
  1000 步输出一帧。自定义模板可以复用该变量，也可以显式固定间隔。
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

需要在飞书群中接收每轮完成和流程终态时，在 `project.yaml` 直接配置自定义
机器人：

```yaml
notifications:
  feishu:
    webhook: https://open.feishu.cn/open-apis/bot/v2/hook/REPLACE
    secret: REPLACE
    timeout_seconds: 5
```

`neptrain doctor --project project.yaml` 会真实发送一条测试消息，同时验证网络、
签名和飞书响应。Controller 运行时只把通知事件放入后台线程；网络超时、签名失败、
飞书拒绝或通知状态文件异常均不会改变 workflow 状态、退出码或科学 ledger。
每个已接受 generation 报告采样、选样、标注、训练集变化和验证 RMSE；流程完成、
失败、评估拒绝、停滞或预算耗尽另发终态消息。投递去重和结果保存在
`.neptrain/notifications.json`，`neptrain workflow status` 会显示其健康状态。
每条进度和终态消息都会显示 workflow 名称与绝对路径，便于区分并发运行的任务。

Slurm 分析任务需要随候选池增长保留内存余量时，可以给独立 analysis target
配置有上限的内存阶梯：

```yaml
execution:
  stage_targets:
    analysis: analysis-cpu
  targets:
    analysis-cpu:
      executor: slurm
      partition: cpu
      cpus_per_task: 1
      memory_ladder: [4G, 8G, 16G]
```

第一次提交使用 4G。只有 `select` 因 Slurm OOM 失败时，Controller 才会自动创建
可追踪的新 attempt，并依次使用 8G、16G；达到最后一级仍失败时流程正常进入
`failed`，不会无限重试。不要同时在 `directives` 中再写 `--mem`。

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

选择 `md.backend: gpumd` 时，route 的 `template_path` 指向 GPUMD `run.in`。
NepTrain 支持模板中的 `nve`，并保留模板选择的 `nvt_*`/`npt_*` 方法、耦合
常数、`time_step` 和 dump 间隔，更新本轮模型、初始温度、步数与确定性
velocity seed；对 `npt_ber` 和
`npt_scr` 还会按模板的 isotropic、orthorhombic 或 triclinic 形式写入目标压强
（GPa）。GPUMD 与 LAMMPS 共用轨迹健康检查和失败窗口契约。Spin workflow
仍明确使用 LAMMPS DynSpin。

每条 route 的 `conditions.temperature_path` 是有顺序的温度探路路径。例如
`[300, 500, 700, 900]` 会先验证 300 K，只有通过后才解锁 500 K。
默认全部温度都会跑到最长时长。显式设置 `production_temperatures` 后，未列入
其中的中间温度只做低成本 smoke 探路。

一个温压条件在本档 replica 正常结束并通过轨迹健康与诊断检查后晋级。FPS 覆盖度
独立决定哪些结构需要标注：发现新颖结构会触发标注和重训，但不会迫使健康轨迹反复
停留在同一时长。晋级后，Controller 同时解锁下一个温度和当前生产温度的下一档
时长。模型更新不会把已经完成的 smoke、short 或 long 证据清零，只有最终
production 认证需要绑定当前模型哈希。`progression.replicas` 控制各时长需要的
独立 MD 次数。

所有通过健康检查的 dump 帧都会参与全局选择。FPS 先按精确元素集合分组，只用
相同元素集合的当前 `train.xyz` 做 warm start；每个仍有新颖结构的温压条件先保留
一个锚点，剩余名额完全按全局 novelty 分配。已经学好的低温条件不会再被等额配满，
高温或新解锁条件可以取得大部分预算。FPS 使用结构级 NEP descriptor；可选择原有的
全原子平均，也可选择按元素规约的均值与标准差通道，但都不把它描述成逐原子局域
新颖度。

NepTrain 只在 FPS 前去除训练集已有结构和同一 route 内的完全重复结构，不按固定
stride 抽帧，也不使用候选数量上限提前裁剪。
`sampling.selection.max_selected` 是每个采样轮最多送去 Label Adapter 的结构数，
不是必须填满的配额。`novelty: auto` 只用当前 `train.xyz` 中的同元素结构拟合描述符
中心和逐特征尺度，再从训练集留一最近邻距离估计保守阈值。候选结构只使用该变换，
不参与尺度拟合；因此极端候选不会反向压低阈值。候选低于训练集已有分辨率时不送
DFT。若本轮一个结构也选不到，
workflow 会跳过 Label Adapter 和重训，直接记录覆盖证据并推进下一档采样，而不是
报错或提交空的 VASP/ABACUS 作业。需要固定策略时，可显式设置
`selection_threshold` 和 `completion_threshold`。

多元素体系建议明确选择按元素规约：

```yaml
sampling:
  selection:
    descriptor_reduction: elementwise_mean_std
    max_selected: 100
    novelty: auto
```

`descriptor_reduction` 有两个值：

- `global_mean`：默认值，保留原有的全原子描述符平均；
- `elementwise_mean_std`：按稳定元素顺序拼接每种元素的描述符均值和标准差，
  避免少数元素或不同元素的变化在总平均中被冲淡。

候选结构、完整 `train.xyz`、自动 novelty 阈值与 FPS 使用同一种规约。切换规约后，
旧规约下的绝对 novelty 阈值不能直接复用；`novelty: auto` 会重新按当前训练集估计。

当所有温度和时长都已覆盖时，配置了独立验证且指标通过的 workflow 才会进入
`complete`。没有独立验证的流程会停在 `coverage_exhausted`：它表示当前描述符和
采样路径已找不到覆盖缺口，不等价于模型已经通过独立精度验证。

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
20 个。Controller 只合并正常完成并通过标签校验的结构。任何失败或取消的标注任务
都会保存诊断后跳过，不再重试，也不会阻塞其余任务；最终结果仍按原始 FPS 顺序写入
`selected-labels.xyz`。如果本轮没有任何标注通过，workflow 会进入 `stalled`，
不会用空数据继续训练。

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

当预训练或微调后的大模型直接替代 DFT 时，workflow stage 不变，只切换
Label Adapter：

```yaml
labeling:
  backend: model
  model_path: ./mace-small.model
  model_name: mace-mp-0-small
  runner: neptrain model-worker mace
  device: cuda
  precision: float32
```

`model_path` 会作为内容寻址输入进入 label task；`model_name`、模型 SHA256、
runner、device 和 precision 会写入 `label-provenance.json`；extxyz 帧本身保留
标签来源、后端、模型名称和模型 SHA256。runner 在
`execution.stage_targets.labeling` 指定的环境执行，因此 Teacher 框架的
PyTorch/CUDA 依赖不需要安装到 Controller 环境。

MACE 适配器由 `NepTrain[mace]` 提供，要求本地 checkpoint 文件和正体积周期
晶胞；它输出 energy、forces 和按 `-stress × volume` 转换的 virial。MACE
不输出磁力，所以这个 runner 不支持 spin workflow。

DeepMD 适配器由 `NepTrain[deepmd]` 提供。DPA-3 多任务模型可写成：

```yaml
labeling:
  backend: model
  model_path: ./DPA-3.2-5M.pt
  model_name: dpa-3.2-5m-omol25
  runner: neptrain model-worker deepmd --head OMol25
  device: cuda
  precision: float32
```

DPA-4 使用相同 runner 和本地 `.pt2` 模型；不新增 workflow backend。模型格式、
head 和推理后端由 DeePMD-kit 处理，NepTrain 仍按本地文件内容记录 SHA256。
当前 DPA-4 要求 DeePMD-kit 3.2 或更新版本。DeepMD runner 同样不生成
`mforce`，因此不支持 spin workflow。

TACE runner 复用官方 `tace-eval` 批量推理，并把它写出的预测 extxyz 归一成
NepTrain 标签：

```yaml
labeling:
  backend: model
  model_path: ./TACE-OAM-7M.pt
  model_name: TACE-OAM-7M
  runner: neptrain model-worker tace --fidelity-index 0
  device: cuda
  precision: float32
```

`--fidelity-index` 会写入传给 TACE 的临时输入结构；省略时保留模型或输入的默认
fidelity。NepTrain 要求模型至少输出 energy、forces，以及 stress 或 virial。
stress 会按 `-stress × volume` 转为 virial，TACE 直接输出的 virial 则原样保留。
普通 TACE checkpoint 不能标注 spin 数据；只有真实输出
`noncollinear_magnetic_forces` 的模型才会被映射为标准 `mforce`。

这里的 `model-worker` 是 label stage 的内部协议，不是独立产品入口；用户仍通过
`neptrain label` 或 workflow 启动标注。

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

默认状态页优先显示当前代次、采样温度路径、实际 MD 进度和历代验证精度。例如：

```text
NepTrain · Fe-spin
路径：/work/neptrain/Fe-spin
状态：运行中 | 第 3/6 代 | 采样中
更新：14:32:08（8 秒前）

采样进度：
300 K ✓ → 500 K ● 3.2/10 ps（2/4 条轨迹完成）→ 700 K ○

验证集精度：
代    状态    E/eV          F/meV·Å⁻¹    V/eV          M/meV/μB    验收
G1    完成    0.0180        210           0.0410        168          通过
G2    完成    0.0142 ↓21%   176 ↓16%      0.0350 ↓15%  149 ↓11%     通过
G3    采样中  -             -             -             -            等待
```

ps 进度来自 MD 已写出的实际 step 和模板中的有效 timestep，不使用墙钟时间估算；
远端文件暂时不可见时会明确显示“ps 暂不可读”。未配置独立验证集时，状态页不把
训练误差冒充为泛化精度，而是显示“暂无可比较数据”。

`--jobs` 按“代次 + 阶段 + attempt”压缩同一批任务。即使同时运行 20 个 MD 或
100 个 DFT 标注任务，也只显示每批的完成、运行、等待和失败计数；逐任务结构仍
完整保留在 `--json` 输出中。

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
- `degraded`：临时 SSH/scheduler 查询失败，Controller 保留原 handle 并重试；
  连接恢复后自动继续，不会把传输故障冒充为计算失败。只有 scheduler 明确返回
  失败、取消或确认任务丢失时，才进入 `failed`。
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
`calculation` 软链指向真实执行目录，便于直接排查。训练完成后会用 Matplotlib
从 `loss.out` 自动生成 `training-convergence.png` 和可审计的
`training-report.json`。配置独立验证集时，evaluate 还会生成按验收阈值归一化的
`evaluation-metrics.png` 与 `evaluation-report.json`；图中 1× 线就是配置阈值。
同一轮预测还会生成 Energy、Force、Virial 的 reference/prediction parity 图
`evaluation-parity.png`，spin 模型会增加 magnetic-force 面板。对应报告记录
validation/model hash、总点数、实际绘制点数和 RMSE；大数组只对显示点做确定性
抽样，RMSE 仍使用全部有限数据。

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
