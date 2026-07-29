# 独立步骤

NepTrain 的训练、MD 和标注命令既可在本机运行，也可通过 `project.yaml` 中的
target 提交到 Slurm。上传、提交、等待和收集进度写到 stderr；最终结果默认是
适合人阅读的摘要。

使用 `--project` 时，backend、模板、温度、压强和步数默认读取项目配置；命令行参数
只覆盖本次需要改变的值。

## 训练

```bash
neptrain train train.xyz \
  --backend torchnep \
  --config nep.in \
  -o nep.txt
```

加入 target 后提交 GPU 作业：

```bash
neptrain train train.xyz \
  --backend torchnep \
  --config nep.in \
  --project project.yaml \
  --target v100 \
  -o nep.txt
```

## 批量 MD

```bash
neptrain md structures/ \
  --backend lammps \
  --model nep.txt \
  --temperature 300 500 700 \
  --steps 100000 \
  --project project.yaml \
  --target cpu \
  --max-concurrent 12 \
  -o trajectories.xyz
```

Slurm target 会将“结构 × 温度”展开成带并发上限的 job array。

GPUMD 使用同一个命令面：

```bash
neptrain md structures/ \
  --backend gpumd \
  --model nep.txt \
  --temperature 300 500 700 \
  --ensemble npt \
  --pressure 0 \
  --steps 100000 \
  --seed 20260727 \
  -o trajectories.xyz
```

不提供模板时会生成可直接运行的 NVE/NVT/NPT 输入。提供 `--template run.in` 时，
ensemble、thermostat/barostat 类型、耦合常数、`time_step` 和 dump 间隔来自
模板；NepTrain 更新模型、初始温度、NPT 目标压强、步数和种子。GPUMD 压强单位
为 GPa。NVE 仍使用 `--temperature` 初始化速度，但不在运行中控温。
两种 backend 都会输出 `trajectory-health.json` 并标注稳定段、炸前帧和坏尾帧；
GPUMD 非零退出但已有完整 dump 帧时也会回收这些帧。Spin MD 只支持 LAMMPS
DynSpin。

## 手动采样

```bash
neptrain select md-300.xyz md-600.xyz \
  --base train.xyz \
  --nep nep.txt \
  --backend auto \
  --max-selected 64 \
  --min-novelty 0.01 \
  --out selected.xyz \
  --report selected.selection.json
```

该命令与自动 workflow 共用同一套层级 FPS：先按精确元素集合分组，再按组规模的
平方根分配初始名额，并在组内平衡轨迹来源、route、温度和压强。`--base` 中只有
元素集合相同的结构会参与对应组的 warm start。`--min-novelty` 是归一化描述符
空间中的严格阈值；精确重复点在阈值为 `0` 时也不会重复入选。

候选结构先按稳定的 structure ID 去重。提供 `--nep` 时通过 NEPAdapters 计算
NEP 描述符；未提供时使用 SOAP（需安装 `NepTrain[soap]`），可用 `--r-cut`、
`--n-max` 和 `--l-max` 调整参数。
需要先过滤异常短键时使用 `--filter 0.6`，并可通过 `--rejected-out` 单独保存被拒
结构。选择报告默认写在输出文件旁，也可由 `--report` 指定路径。

## 批量标注

```bash
neptrain label candidates.xyz \
  --backend vasp \
  --input-file INCAR \
  --resources /shared/potpaw_PBE \
  --potcar-manifest vasp-resources.json \
  --project project.yaml \
  --target label \
  --structures-per-job 1 \
  --max-concurrent 20 \
  -o labeled.xyz
```

`--resources` 只给出资源根目录，不足以锁定赝势版本。VASP 还必须提供
`--potcar-manifest`；ABACUS 必须提供 `--resource-manifest`。manifest 逐文件记录
相对路径和 SHA256，VASP 还记录精确 `TITEL`、family 和 release。例如：

```json
{
  "protocol": "neptrain.vasp-resources.v1",
  "family": "PBE",
  "release": "2025-01",
  "elements": {
    "Fe": {
      "path": "Fe_pv/POTCAR",
      "sha256": "<64 hex characters>",
      "titel": "PAW_PBE Fe_pv 02Aug2007"
    }
  }
}
```

本地 target 在 prepare 时校验真实文件；远程 target 必须在
`execution.targets.<name>.labeling_resource_path` 写目标机绝对路径，并先运行
`neptrain doctor --project project.yaml`。worker 在真正启动 VASP/ABACUS 前会再校验一次。
VASP 路径只接受 `Fe/POTCAR` 或 `Fe_pv/POTCAR` 这样的单层 setup 目录，并由
NepTrain 显式写入 ASE `setups`；不会校验 manifest 中的一个文件，却让 ASE
按默认规则读取另一个文件。
VASP 的 `ISPIN=2` 只表示共线自旋极化电子计算，结果仍是普通
energy/force/virial 标签并记录 `dft_electronic_mode`；它不会生成
`spin/mforce`。非共线、SOC 或真正的 spin-force 标注必须使用 ABACUS DeltaSpin。

预训练或微调后的等变模型可以作为与 VASP/ABACUS 平级的 Label Adapter：

```bash
neptrain label candidates.xyz \
  --backend model \
  --model mace-small.model \
  --model-name mace-mp-0-small \
  --runner 'neptrain model-worker mace' \
  --device cuda \
  --precision float32 \
  --project project.yaml \
  --target teacher-gpu \
  -o labeled.xyz
```

`model-worker mace` 的运行时由 `NepTrain[mace]` 提供，读取本地 MACE checkpoint。
它复用一个 ASE calculator 批量计算结构，将 ASE stress 转成
`virial = -stress × volume`，并保持结构顺序、晶胞和 PBC 不变。当前 MACE
runner 不输出磁力，所以明确拒绝含 `spin` 的输入。

DeepMD 使用同一个 runner 覆盖 DPA-3、DPA-4 和其它 DeepMD 模型：

```bash
neptrain label candidates.xyz \
  --backend model \
  --model DPA-3.2-5M.pt \
  --model-name dpa-3.2-5m-omol25 \
  --runner 'neptrain model-worker deepmd --head OMol25' \
  --device cuda \
  --precision float32 \
  -o labeled.xyz
```

模型格式和后端由 DeePMD-kit 识别。多任务模型通过 runner 的 `--head` 选择
分支；DPA-4 仍使用这个 runner，只需把 `--model` 换成本地 `.pt2` 文件。
`model-worker deepmd` 的运行时由 `NepTrain[deepmd]` 提供；DPA-4 需要支持该格式的
DeePMD-kit 3.2 或更新版本。

TACE 使用官方 `tace-eval` 批量预测接口：

```bash
neptrain label candidates.xyz \
  --backend model \
  --model TACE-OAM-7M.pt \
  --model-name TACE-OAM-7M \
  --runner 'neptrain model-worker tace --fidelity-index 0' \
  --device cuda \
  --precision float32 \
  -o labeled.xyz
```

TACE 先写临时预测 extxyz，NepTrain 再校验并归一化
`TACE_energy`、`TACE_forces` 和 stress/virial。含 `spin` 的输入还必须得到
`TACE_noncollinear_magnetic_forces`，否则任务失败，不会补零。

所有模型 runner 都遵守五参数协议：`--model`、`--input`、`--output`、
`--device` 和 `--precision`。runner 必须输出顺序不变的规范 extxyz。
NepTrain 负责模型 hash、结构身份、energy/forces/virial、可选 spin/mforce
和最终原子发布；runner 失败或标签不完整时不会产生部分结果。完整示例见仓库的
`examples/distillation-mace/`、`examples/distillation-deepmd/` 和
`examples/distillation-tace/`。
`model-worker` 只承载 NepTrain 内部 runner 协议，不是需要用户单独学习的命令；
正常使用请始终从 `neptrain label` 进入。

任务提交后通过统一 task 命令管理：

```bash
neptrain task status runs/label-...
neptrain task logs runs/label-...
neptrain task wait runs/label-...
neptrain task retry runs/label-...
neptrain task cancel runs/label-...
```

- `status` 只做一次有界查询和收集，不会永久等待。
- `wait` 持续轮询到终态，并在状态变化时把进度写到 stderr。
- `cancel` 取消当前 Slurm attempt，不删除已完成且校验通过的 shard。
- `retry` 归档旧 metadata，创建新 attempt，只提交失败、取消、缺失或损坏的
  shard。
- `logs` 获取日志，不改变 operation 或 scheduler 状态。

作业从 `prepared/submitted/running/cancelling` 进入
`complete/failed/cancelled/damaged`。作业已从 `squeue` 消失时会继续查询
`sacct`；在有界 accounting grace 之后仍不可见才标记 `LOST` 并允许 retry，
不会无限显示 RUNNING。

手动任务目录只保留两层用户可见结构：

```text
label-50/
├── labeled.xyz -> ../50-labeled.xyz
├── operation.json
├── remote.txt
├── job.sbatch
├── logs/
└── jobs/
    └── 000000/
        ├── input.xyz
        ├── request.json
        ├── execution.json
        ├── result.json
        ├── labeled.xyz
        └── calculation/
```

根目录的 `labeled.xyz`、`trajectory.xyz` 或 `nep.txt` 始终直达合并后的最终
结果；`remote.txt` 记录跨平台任务的主机和真实计算路径。每个 array job 的
输入、状态和结果直接位于 `jobs/<index>/`，后端原始输出集中在其
`calculation/` 中。调度器输出统一放在 `logs/`。

提交后会立即显示任务状态、作业号、运行目录和下一条建议命令。脚本需要稳定的
机器输出时，给训练、MD、label 或 task 子命令加 `--json`：

```bash
neptrain label candidates.xyz --project project.yaml --target label --json
neptrain task status runs/label-... --json
```

上传、提交、排队、运行、收集和合并进度只写 stderr。`--json` 的 stdout 只有
一个对象；状态使用 `neptrain.manual-status.v1`，日志使用
`neptrain.manual-logs.v1`。字段 `state`、`completed`、`failed`、`total`、
`jobs`、`errors`、`result` 和 `next_action` 是面向脚本的接口。

只有全部 shard 成功后才会发布最终输出；已有输出默认拒绝覆盖，确认替换时使用
`--force`。
同一集群共享文件系统上的 array 会自动完成发布；跨平台任务由 `task status` 或
`task wait` 同步并发布。一次状态检查会把尚未完整落盘的 shard 结果合并成一个
归档传回，因此任务数增加时不会逐个建立 SSH/SCP 连接；若上次同步中断，
再次执行相同命令会自动补齐缺少的 `result.json` 和实际结果文件。`task logs`
也以一次归档传回当前作业的全部调度器日志。

平台程序默认从 `PATH` 查找。若平台要求 `srun vasp_std` 等命令，把
`NEPTRAIN_VASP_COMMAND`（或对应的 `NEPTRAIN_ABACUS_COMMAND`、
`NEPTRAIN_NEP_COMMAND`、`NEPTRAIN_GPUMD_COMMAND`）写入 target 的
`environment`。

`operation.json` 是手动任务身份、输入 manifest、shard 顺序和 attempt 列表的
权威；`jobs/<index>/execution.json` 只记录运行态，`result.json` 固定所属
operation、结构顺序和 artifact hash。最终文件只在全部 shard 均通过该契约后
原子发布。损坏 `execution.json`、缺少 `result.json` 或顺序不一致都会显示为
可诊断失败，不会把部分 `labeled.xyz` 当成完整结果。
