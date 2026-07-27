# 从 VASP 开始跑一代 NepTrain workflow

这个教程面向第一次使用 NepTrain 的 Slurm 用户。完成后你会得到一条真实的：

```text
初始数据 → Student 训练 → GPUMD 采样 → FPS → VASP 标注
        → 合并数据 → Student 重训 → 验收
```

本例使用 Al，VASP 负责新结构的真实标注。仓库不能分发 VASP 或 POTCAR；你需要
有可用的 `vasp_std` 和合法的 PAW_PBE 资源。

## 1. 进入示例并安装环境

从仓库根目录执行：

```bash
cd examples/workflow-vasp-slurm
pip install 'NepTrain[torchnep]'
```

还需要让训练节点能运行 `neptrain` 和 TorchNEP，让 MD 节点能运行 `gpumd`，
让 VASP 节点能运行 `vasp_std`。先记录三个节点各自使用的 conda/module 设置，
后面写进 `env-*.sh`。

## 2. 生成教程初始数据

```bash
python ../prepare_al_seed.py --output-dir .
```

脚本会生成：

| 文件 | 用途 |
|---|---|
| `train.xyz` | 24 个带 energy/forces/virial 的初始训练结构 |
| `validation.xyz` | 4 个独立的教程验收结构 |
| `structures/al.xyz` | GPUMD 的起始结构 |

这些标签来自 ASE EMT，只用于验证 workflow 机械过程。正式计算必须把
`train.xyz` 和 `validation.xyz` 换成与你的 VASP 设置一致的数据，否则不同理论
水平会被混进同一个训练集。

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

## 5. 运行预检

```bash
neptrain doctor \
  --project project.yaml \
  --training-backend torchnep \
  --md-backend gpumd
```

预检会检查 schema-v8 配置、Slurm 命令、setup script 和计算节点上的 POTCAR
路径及 SHA256。它不能替代一次真实的 VASP 计算；许可证、MPI 和 VASP 本身是否
能启动，仍应由下一步确认。

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
| `generations/0001/md/` | GPUMD 轨迹和健康报告 |
| `generations/0001/select/` | FPS 选择结果和选择报告 |
| `generations/0001/label/selected-labels.xyz` | VASP 新标签 |
| `generations/0001/label/label-provenance.json` | VASP 输入和资源来源 |
| `generations/0001/dataset/` | 合并后的训练集 |
| `generations/0001/retrain/` | 重训模型和训练曲线 PNG |
| `generations/0001/evaluate/` | 验收结果和评估图 PNG |

本例把 `max_model_generations` 设为 `1`。状态最后出现 `budget_exhausted` 表示
教程设定的一代预算已经用完，不等于 Slurm 或 VASP 失败。真正失败会在 stage/job
状态和对应日志中显示。

## 9. 换成正式项目

正式使用前必须修改：

- 用同一 VASP 理论水平的数据替换 EMT `train.xyz`/`validation.xyz`。
- 按目标体系扩展 manifest 中的全部元素。
- 检查 `ENCUT`、赝势、k 点、电子收敛和自旋设置。
- 把教程的 10–80 步 NVE 改成经过稳定性验证的 NVE/NVT/NPT 路径。
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
