# 自动迭代

## 创建项目

```bash
neptrain workflow init --profile slurm --directory fe-project
cd fe-project
neptrain doctor --project project.yaml
```

项目只接受 `schema_version: 5`。温度、压强和 MD 步数统一位于 `sampling`，不再保留
旧字段或运行期迁移。

当前配置职责如下：

- `md`：选择 LAMMPS/GPUMD、起始结构、用户模板和运行命令。
- `sampling`：温度、压强、场景递进、轨迹健康检查和 FPS 抽样上限。
- LAMMPS 模板：`timestep`、恒温/恒压阻尼、spin 积分参数和 dump 频率。
- `execution.targets.*.setup_script`：module、Python 环境和 LAMMPS plugin。

NepTrain 只向 LAMMPS 模板注入温度、压强、步数、独立 replica 的确定性随机种子
`{{ seed }}`，以及模型/结构/输出路径，不管理 `plugin_path`。阻尼、积分器和
spin 参数仍由用户模板决定。

`sampling.conditions.temperature_path` 是有顺序的温度探路路径。例如
`[300, 500, 700, 900]` 会先验证 300 K，只有通过后才解锁 500 K。
`production_temperatures` 是必须跑到最长时长的工作温度；中间温度默认只做
低成本 smoke 探路，避免把所有温度和所有时长做笛卡尔积。

场景通过后，Controller 会同时解锁下一个温度和当前生产温度的下一档时长。
失败场景保留在原位置，采集稳定段和炸前帧，经 FPS、DFT 和重训后重试，不会
越过失败温度。`progression.replicas` 控制各时长需要的独立 MD 次数，默认在
long 和 production 阶段增加重复数。

所有通过健康检查的 dump 帧都会参与全局 FPS。NepTrain 只在 FPS 前去除训练集
已有结构和完全重复结构，不按固定 stride 抽帧，也不使用候选数量上限提前裁剪。
`sampling.selection.max_selected` 是每个采样轮最多送去 DFT 的结构数。常规标注
下限由系统自动取其一半，例如上限 100 时优先积累至少 50 个；若当前场景 frontier
已经耗尽，或出现物理失败需要抢救，则允许较小批次提前提交。

这里的“一代”严格绑定一个模型哈希。同一代可以连续运行多个 MD wave，但所有
候选必须由同一个模型产生；候选池 manifest 和每个 extxyz 帧都会记录
`sampling_model_sha256`。诊断超过阈值或 MD 发生物理失败时才更新模型；新模型
完成 evaluate 并写入 lineage
后，下一轮 MD 才能开始。若当前模型已经通过新 DFT 诊断，则保持同一模型继续认证，
避免每次改模型都把长 MD 证据清零。旧模型轨迹不能混入新模型候选池。

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

`workflow.max_iterations` 是最大采样轮预算，不是成功条件。所有生产温度、最长
时长、replica、轨迹 DFT 诊断和全局 validation 同时通过后会提前结束；预算用尽
但仍未收敛时状态为 `budget_exhausted`，连续两轮没有新覆盖或模型改进时状态为
`stalled`，都不会伪装成 `complete`。

默认停止只退出 Controller，保留当前计算任务：

```bash
neptrain workflow stop fe-workflow
```

确认整个流程已经作废时，可同时取消当前 process 或 Slurm 作业：

```bash
neptrain workflow stop fe-workflow --cancel-jobs
```

取消动作会记录到 workflow 历史；后续恢复会创建新的 stage attempt。
