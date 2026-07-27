# 从 ABACUS 开始跑一代 NepTrain workflow

这个教程面向第一次使用 NepTrain 的 Slurm 用户。完成后你会跑通：

```text
初始数据 → Student 训练 → GPUMD 采样 → FPS → ABACUS 标注
        → 合并数据 → Student 重训 → 验收
```

本例使用 Al 和 ABACUS 平面波基组。仓库不分发 ABACUS 赝势；你需要有可运行的
`abacus` 和对应的 UPF 文件。

## 1. 进入示例并安装环境

```bash
cd examples/workflow-abacus-slurm
pip install 'NepTrain[torchnep]'
```

训练节点需要 TorchNEP，MD 节点需要 GPUMD，标注节点需要 ABACUS。把三类节点
使用的 conda/module 设置分别写入 `env-training.sh`、`env-gpumd.sh` 和
`env-abacus.sh`。

## 2. 生成教程初始数据

```bash
python ../prepare_al_seed.py --output-dir .
```

会生成 `train.xyz`、`validation.xyz` 和 `structures/al.xyz`。前两个文件使用
ASE EMT 标签，只用于验证 workflow 机械过程。正式计算必须换成与 ABACUS
赝势、泛函、基组和 k 点设置一致的第一性原理数据。

## 3. 固定 ABACUS 资源

复制 manifest 模板：

```bash
cp abacus-resources.example.json abacus-resources.json
```

假设你的资源根目录是 `/shared/abacus-resources`，Al 赝势文件是
`Al_ONCV_PBE.upf`：

```bash
sha256sum /shared/abacus-resources/Al_ONCV_PBE.upf
grep -im1 "element" /shared/abacus-resources/Al_ONCV_PBE.upf
```

macOS 上将 `sha256sum` 换成 `shasum -a 256`。把真实文件名、哈希和资源发行版
写进 `abacus-resources.json`：

```json
{
  "elements": {
    "Al": {
      "orbital": null,
      "pseudopotential": {
        "path": "Al_ONCV_PBE.upf",
        "sha256": "<上一步得到的 64 位小写哈希>"
      }
    }
  },
  "protocol": "neptrain.abacus-resources.v1",
  "release": "<你的资源发行版名称>"
}
```

当前 `INPUT` 使用 `basis_type pw`，所以 `orbital` 可以是 `null`。改成
`basis_type lcao` 后，必须同时提供 `.orb` 的相对路径和 SHA256；否则
NepTrain 会在启动 ABACUS 前拒绝任务。

## 4. 修改集群配置

打开 `project.yaml` 并替换：

1. `REPLACE_GPU_PARTITION`：训练和 GPUMD 分区。
2. `REPLACE_CPU_PARTITION`：ABACUS 分区。
3. 两处 `/REPLACE/WITH/YOUR/abacus-resources`：登录节点和计算节点看到的资源
   绝对路径。
4. 三个 `env-*.sh` 中的 conda/module 命令。

示例使用 `NEPTRAIN_ABACUS_COMMAND: srun abacus`。如果集群采用其它 launcher，
只修改这个环境变量。

`INPUT` 已包含 `kspacing 0.25`，所以 `kpoint_mode: auto` 会沿用它。检查是否
还有占位符：

```bash
grep -R "REPLACE" project.yaml env-*.sh abacus-resources.json
```

没有输出才继续。

## 5. 运行预检

```bash
neptrain doctor \
  --project project.yaml \
  --training-backend torchnep \
  --md-backend gpumd
```

预检会检查 schema、Slurm target、setup script，以及计算节点上的 UPF 路径和
SHA256。它不会消耗一次真实 ABACUS 计算，也不能验证许可证/MPI/运行时组合。

## 6. 先标注一个结构

```bash
neptrain label structures/al.xyz \
  --backend abacus \
  --project project.yaml \
  --target abacus \
  --wait \
  --output abacus-check.xyz
```

成功后检查三类标签：

```bash
python - <<'PY'
from ase.io import read
atoms = read("abacus-check.xyz")
print("energy:", atoms.get_potential_energy())
print("max |force|:", abs(atoms.get_forces()).max())
print("virial shape:", atoms.info["virial"].shape)
PY
```

失败时用 `neptrain task logs <命令输出的 run_directory>` 查看 ABACUS 和 Slurm
日志。不要在独立标注失败时继续启动 workflow。

## 7. 准备并启动 workflow

```bash
neptrain workflow run project.yaml --prepare-only
neptrain workflow run abacus-tutorial-workflow
neptrain workflow status abacus-tutorial-workflow --jobs
```

要停止 controller 并取消当前任务：

```bash
neptrain workflow stop abacus-tutorial-workflow
```

## 8. 跑完检查什么

| 位置 | 内容 |
|---|---|
| `generations/0001/md/` | GPUMD 轨迹和健康报告 |
| `generations/0001/select/` | FPS 选择结果 |
| `generations/0001/label/selected-labels.xyz` | ABACUS 新标签 |
| `generations/0001/label/label-provenance.json` | INPUT、UPF/ORB 哈希和后端来源 |
| `generations/0001/dataset/` | 合并后的训练集 |
| `generations/0001/retrain/` | 重训模型和训练曲线 PNG |
| `generations/0001/evaluate/` | 验收结果和评估图 PNG |

示例只允许一代，所以最后的 `budget_exhausted` 表示教程预算用完，并不等于
ABACUS 失败。真正失败应结合 stage/job 状态和日志判断。

## 9. 换成正式项目

- 用同一 ABACUS 理论水平的数据替换 EMT 初始数据。
- 为所有元素固定 UPF；LCAO 还要固定 ORB。
- 检查 `ecutwfc`、k 点、smearing、磁性和 SCF 收敛设置。
- 将 10–80 步 NVE 换成经过目标体系验证的采样路径。
- 增加训练/验证规模和 workflow 代数。
- 按集群资源调整 Slurm target 与标注并发。

## 常见问题

| 现象 | 常见原因 | 处理 |
|---|---|---|
| `resource hash mismatch` | UPF/ORB 与 manifest 不一致 | 重新计算 SHA256 |
| `does not declare its element` | UPF 缺少可识别的 `element=` 元数据 | 使用正确的 ABACUS UPF 或检查文件 |
| `missing orbital` | `basis_type lcao` 但 manifest 没有 ORB | 补 `.orb` 路径和哈希 |
| `execution target ... FAIL` | setup script 或分区仍是占位值 | 清理全部 `REPLACE` |
| ABACUS job 很快退出 | module、MPI launcher 或 INPUT 不适配集群 | 查看 task/stage 日志 |
| 轨迹出现 NaN | 初始 Student 或 MD 参数不稳定 | 看健康报告，先缩短步长/温度并补初始数据 |
