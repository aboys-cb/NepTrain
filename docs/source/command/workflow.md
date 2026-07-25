# 自动迭代

## 创建项目

```bash
neptrain workflow init --profile slurm --directory fe-project
cd fe-project
neptrain doctor --project project.yaml
```

项目只接受 `schema_version: 7`。自动采样只读取 `sampling.routes`，不再接受
全局 `md.structures`、`md.template_path`、`sampling.conditions` 或
`sampling.progression`，也不在运行时迁移旧配置。

当前配置职责如下：

- `md`：选择 LAMMPS/GPUMD 和推理后端。
- `sampling.routes`：每条采样路径显式绑定结构、模板、条件和递进策略。
- `sampling.candidate_pool` / `sampling.selection`：轨迹健康检查和 FPS 抽样上限。
- LAMMPS 模板：物理过程、`timestep`、阻尼、spin 积分参数和 dump 频率。
- `execution.targets.*.setup_script`：module、Python 环境和 LAMMPS plugin。

一个 workflow 可以配置多条 route：

```yaml
schema_version: 7

md:
  backend: lammps
  spin: false

sampling:
  routes:
    - id: route_a
      structures: [./structures/a.xyz]
      template_path: ./templates/a.in
      conditions:
        temperature_path: [300, 600, 900]
        production_temperatures: [300, 900]
        pressure: 0.0
      progression:
        steps:
          smoke_passed: 5000
          short_stable: 25000
          long_stable: 250000
          production_ready: 1000000
        replicas:
          smoke_passed: 1
          short_stable: 1
          long_stable: 2
          production_ready: 3

    - id: route_b
      structures: [./structures/b.xyz]
      template_path: ./templates/b.in
      conditions:
        temperature_path: [500, 1000, 1500]
        production_temperatures: [500, 1500]
        pressure: 0.0
      progression:
        steps:
          smoke_passed: 5000
          short_stable: 25000
          long_stable: 250000
          production_ready: 1000000
        replicas:
          smoke_passed: 1
          short_stable: 1
          long_stable: 2
          production_ready: 3

  candidate_pool:
    pre_failure_frames: 2
    bad_tail_frames: 1
    health: {}
  selection:
    max_selected: 100
    novelty: auto
```

一个 workflow 会调度全部 route。同一个结构可以显式出现在多条 route 中；
`route_id` 和内容指纹会隔离场景成熟度与恢复结果。

默认情况下，全部 route 使用 `execution.stage_targets.sampling`。需要把某条
route 送到另一台 Slurm 平台时，只写例外映射：

```yaml
execution:
  stage_targets:
    training: train
    sampling: md-default
    labeling: dft
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
`production_temperatures` 是必须跑到最长时长的工作温度；中间温度默认只做
低成本 smoke 探路，避免把所有温度和所有时长做笛卡尔积。

场景通过后，Controller 会同时解锁下一个温度和当前生产温度的下一档时长。
失败场景保留在原位置，采集稳定段和炸前帧，经 FPS、DFT 和重训后重试，不会
越过失败温度。`progression.replicas` 控制各时长需要的独立 MD 次数。

所有通过健康检查的 dump 帧都会参与全局选择。FPS 先按精确元素集合分组，按
组大小平方根分配初始名额；组内再按 route、温压条件和轨迹窗口做软平衡，并且
只用相同元素集合的训练结构 warm-start。未用完的名额会确定性转给仍有新颖候选
的元素组。当前使用结构级 NEP descriptor，不把它描述成逐原子局域新颖度。

NepTrain 只在 FPS 前去除训练集已有结构和同一 route 内的完全重复结构，不按固定
stride 抽帧，也不使用候选数量上限提前裁剪。
`sampling.selection.max_selected` 是每个采样轮最多送去 DFT 的结构数。常规标注
下限由系统自动取其一半，例如上限 100 时优先积累至少 50 个；若当前场景 frontier
已经耗尽，或出现物理失败需要抢救，则允许较小批次提前提交。

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

## DFT k 点

默认让用户输入文件管理 k 点：

```yaml
dft:
  backend: vasp
  input_path: ./INCAR
  resource_path: /shared/potpaw_PBE
  kpoint_mode: auto
```

`auto` 会优先保留 VASP INCAR 中的 `KSPACING`/`KGAMMA`，或 ABACUS INPUT
中的 `kspacing`。输入中没有这些设置时，适配器才按默认参数生成网格。

确实需要 NepTrain 接管时再显式配置：

```yaml
# 按间距设置
dft:
  kpoint_mode: kspacing
  kspacing: 0.2
  gamma_centered: true

# 或按网格密度参数设置
dft:
  kpoint_mode: kpoints
  kpoints: [4, 4, 4]
  gamma_centered: true
```

`kspacing` 和 `kpoints` 不能同时配置。手动命令保持同样规则：
`neptrain dft --kspacing 0.2 ...` 或 `neptrain dft --ka 4,4,4 ...`；
两者互斥。当前输入接口读取 INCAR/ABACUS INPUT 内的 k 点间距，不把同目录下
单独的 VASP `KPOINTS` 或 ABACUS `KPT` 文件作为自动 workflow 输入。

## 准备和运行

```bash
neptrain workflow run project.yaml --prepare-only
neptrain workflow run project.yaml
```

输出目录默认使用 `workflow.id`。Controller 默认脱离终端，按 ledger 推进：

```text
train → explore → select → label → diagnose → merge → retrain → evaluate
```

训练、MD 和 DFT stage 使用与独立命令相同的 Adapter 和 execution target。

## 状态与恢复

```bash
neptrain workflow status fe-workflow
neptrain workflow status fe-workflow --jobs
neptrain workflow resume fe-workflow
neptrain workflow stop fe-workflow
neptrain workflow extend fe-workflow 5
```

Controller 不依赖 Slurm `afterok`。失败后只重跑 ledger 中未完成的阶段；已完成
workflow 重复运行是安全 no-op。

`workflow.max_model_generations` 是最大模型代数预算，不是成功条件。配置 `evaluation`
时，所有生产温度、最长时长、replica、轨迹 DFT 诊断和独立 validation 同时通过
后才会提前结束。预算用尽但仍未收敛时状态为 `budget_exhausted`，连续两轮没有
新覆盖或模型改进时状态为 `stalled`，都不会伪装成 `complete`。

`evaluation` 可以整块省略。此时 workflow 仍可采样、标注、重训和记录场景证据，
但 `validation_accepted` 保持为空，流程不会伪造 validation passed，也不会把
该模型标成已完成独立验证。

默认停止只退出 Controller，保留当前计算任务：

```bash
neptrain workflow stop fe-workflow
```

确认整个流程已经作废时，可同时取消当前 process 或 Slurm 作业：

```bash
neptrain workflow stop fe-workflow --cancel-jobs
```

取消动作会记录到 workflow 历史；后续恢复会创建新的 stage attempt。
