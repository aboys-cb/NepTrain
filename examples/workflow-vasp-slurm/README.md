<div align="center">
<a href="README.en.md">English</a> | <strong>简体中文</strong>
</div>

# 从 VASP 开始跑一代 NepTrain workflow

这个教程面向第一次使用 NepTrain 的 Slurm 用户。完成后你会得到一条真实的：

```text
初始数据 → Student 训练 → GPUMD 采样 → FPS → VASP 标注
        → 训练前精度评估 → 合并数据 → 决定继续采样或最终训练
```

本例使用 Al，VASP 负责新结构的真实标注。仓库不能分发 VASP 或 POTCAR；你需要
有可用的 `vasp_std` 和合法的 PAW_PBE 资源。

## 1. 进入示例并安装环境

从仓库根目录执行：

```bash
cd examples/workflow-vasp-slurm
python -m pip install -e '../..[torchnep]'
```

从当前源码安装，确保 NepTrain 与附带的占位符模板匹配；MD 计算节点也应使用同一
源码版本。`gpumd-npt.in` 由 NepTrain 渲染后运行，不要直接交给 GPUMD。

还需要让训练节点能运行 `neptrain` 和 TorchNEP，让 MD 节点能运行 `gpumd`，
让 VASP 节点能运行 `vasp_std`。先记录三个节点各自使用的 conda/module 设置，
后面写进 `env-*.sh`。

## 2. 生成未标注的种子结构

```bash
python ../prepare_al_seed.py --output-dir .
```

生成 `seed-train.xyz`（24 帧）、`seed-validation.xyz`（4 帧可选测试结构）和
`structures/al.xyz`。此时没有能量、力和 virial 标签，也不会生成 EMT 标签。
第 6 步使用与 workflow 相同的后端、输入和资源清单标注种子，生成 `train.xyz`。

## 3. 固定 POTCAR

先复制 manifest 模板：

```bash
cp vasp-resources.example.json vasp-resources.json
```

假设你的资源根目录是 `/shared/potpaw_PBE`，Al 文件是
`/shared/potpaw_PBE/Al/POTCAR`。读取真实哈希和 `TITEL`：

```bash
sha256sum /shared/potpaw_PBE/Al/POTCAR
grep -m1 TITEL /shared/potpaw_PBE/Al/POTCAR
```

macOS 上将 `sha256sum` 换成 `shasum -a 256`。把结果写进
`vasp-resources.json`：

```json
{
  "elements": {
    "Al": {
      "path": "Al/POTCAR",
      "sha256": "<上一步得到的 64 位小写哈希>",
      "titel": "<TITEL 等号右侧的完整内容>"
    }
  },
  "family": "PAW_PBE",
  "protocol": "neptrain.vasp-resources.v1",
  "release": "<你的 POTCAR 发行版名称>"
}
```

`path` 必须是 `Al/POTCAR` 或 `Al_<setup>/POTCAR`。NepTrain 会用同一条记录
校验文件并驱动 ASE 选择 setup，避免“校验一个 POTCAR、实际计算另一个”。

## 4. 修改集群配置

打开 `project.yaml`，至少替换四类占位值：

1. `REPLACE_GPU_PARTITION`：训练和 GPUMD 使用的 GPU 分区。
2. `REPLACE_CPU_PARTITION`：VASP 使用的 CPU 分区。
3. 两处 `/REPLACE/WITH/YOUR/potpaw_PBE`：登录节点和计算节点看到的 POTCAR
   绝对路径；共享文件系统下通常相同。
4. `env-training.sh`、`env-gpumd.sh` 和 `env-vasp.sh` 中的 conda/module 命令。

示例通过 `NEPTRAIN_VASP_COMMAND: srun vasp_std` 启动 VASP。如果你的集群使用
其它 launcher，只改这个值，不要修改 NepTrain worker。

本例 `INCAR` 已包含 `KSPACING = 0.25` 和 `KGAMMA = True`，因此
`kpoint_mode: auto` 会保留这组设置。

检查文件中是否还剩占位符：

```bash
grep -R "REPLACE" project.yaml env-*.sh vasp-resources.json
```

这条命令没有输出才继续。

## 5. 确认运行环境已补齐

先填写资源路径、哈希、分区和环境脚本，再提交下面的单结构标注。
完整的 `doctor --project` 检查放在种子标注之后；此时 `train.xyz` 尚未生成。
单结构计算负责确认真实后端可运行，不能只用命令存在或资源哈希正确代替。

## 6. 先标注一个结构

在启动自动 workflow 前，先做一次真实后端冒烟：

```bash
neptrain label structures/al.xyz \
  --backend vasp \
  --project project.yaml \
  --target vasp \
  --wait \
  --output vasp-check.xyz
```

成功后 `vasp-check.xyz` 必须能被 ASE 读出能量、力和 virial：

```bash
python - <<'PY'
from ase.io import read
atoms = read("vasp-check.xyz")
print("energy:", atoms.get_potential_energy())
print("max |force|:", abs(atoms.get_forces()).max())
print("virial shape:", atoms.info["virial"].shape)
PY
```

如果这一步失败，不要启动 workflow。先运行
`neptrain task logs <命令输出的 run_directory>` 查看 Slurm stdout/stderr。

单结构通过后，用同一配置标注种子。只有训练集必需；未提供可选测试集时提示并跳过。

```bash
neptrain label seed-train.xyz --backend vasp --project project.yaml --target vasp --wait --output train.xyz
# Optional / 可选：
neptrain label seed-validation.xyz --backend vasp --project project.yaml --target vasp --wait --output validation.xyz
neptrain doctor --project project.yaml
```

## 7. 准备并启动 workflow

先只创建工作目录，方便检查最终配置快照：

```bash
neptrain workflow run project.yaml --prepare-only
```

确认生成了 `vasp-tutorial-workflow/` 后启动 controller：

```bash
neptrain workflow run vasp-tutorial-workflow
```

查看进度：

```bash
neptrain workflow status vasp-tutorial-workflow --jobs
```

停止 controller 并取消它当前管理的任务：

```bash
neptrain workflow stop vasp-tutorial-workflow
```

## 8. 跑完检查什么

一代完成后重点看：

| 位置 | 内容 |
|---|---|
| `generations/0001/explore/` | GPUMD 轨迹和健康报告 |
| `generations/0001/select/` | FPS 选择结果和选择报告 |
| `generations/0001/label/selected-labels.xyz` | VASP 新标签 |
| `generations/0001/label/label-provenance.json` | VASP 输入和资源来源 |
| `generations/0001/update/` | 合并后的训练集 |
| `generations/0001/train/` | 本代采样模型和训练曲线 PNG |
| `generations/0001/evaluate/` | 新增标签的训练前预测误差 |

`evaluation` 只输出辅助测试，不阻止流程继续；采样精度阈值在 `workflow.convergence` 中。
示例阈值仅供教程使用。只有精度、生产覆盖和连续达标要求都满足，才会进入最终训练。
辅助测试报告位于 `generations/0001/train/`。

本例把 `max_model_generations` 设为 `1`。状态最后出现 `budget_exhausted` 表示
教程设定的一代预算已经用完，不等于 Slurm 或 VASP 失败。真正失败会在 stage/job
状态和对应日志中显示。

## 9. 换成正式项目

正式使用前必须修改：

- 扩充与当前 VASP 理论水平一致的训练数据，覆盖目标应变、温度和缺陷。
- 按目标体系扩展 manifest 中的全部元素。
- 检查 `ENCUT`、赝势、k 点、电子收敛和自旋设置。
- 把教程的 10–80 步 NPT 改成经过稳定性验证的 NPT 路径。
- 增加训练规模、验证覆盖和 `max_model_generations`。
- 根据队列策略调整 `structures_per_job`、`max_concurrent` 和 Slurm 资源。

## 常见问题

| 现象 | 常见原因 | 处理 |
|---|---|---|
| `resource hash mismatch` | POTCAR 被替换或 manifest 写错 | 重新计算 SHA256；不要跳过校验 |
| `POTCAR version mismatch` | `titel` 不是等号右侧完整值 | 重新复制 `grep -m1 TITEL` 的值 |
| `execution target ... FAIL` | setup script 或分区仍是占位值 | 先清理全部 `REPLACE` |
| VASP job 很快退出 | module、许可证、MPI launcher 不匹配 | 查看 `neptrain task logs` 或 stage 日志 |
| 轨迹出现 NaN | 初始 Student 或 MD 参数不稳定 | 看 `trajectory-health.json`；先缩短步长/温度并扩充初始数据 |

## NPT 设置与验证边界

`gpumd-npt.in` 使用各向同性 `npt_scr`，目标压强为 0 GPa；100 GPa 是用于压强耦合的
粗略弹性模量参数，温度/压强耦合参数分别为 100/1000 步。这些是教程设置，不是拟合的
材料参数。4 原子晶胞和 10–80 步只检查输入输出与流程，不能证明平衡或势函数可用于生产。
语法见 [GPUMD 系综文档](https://gpumd.org/gpumd/input_parameters/ensemble_standard.html)。
一代预算通常以 `budget_exhausted` 结束，不应预期得到 `complete`。

## 改用普通 LAMMPS NPT

备选配置为 `project-lammps.yaml`，配套 `lammps-npt.in` 和 `env-lammps.sh`，复用同一套
VASP 训练数据与资源。补齐备选 YAML 的分区/路径以及 LAMMPS/NEPAdapters 环境后执行：

```bash
neptrain doctor --project project-lammps.yaml
neptrain workflow run project-lammps.yaml --prepare-only
neptrain workflow run vasp-lammps-tutorial-workflow
```

LAMMPS `units metal` 的压强单位是 **bar**；框架按模板填值，不把 GPUMD 的 GPa 自动换算过来。
