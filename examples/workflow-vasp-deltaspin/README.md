<div align="center"><a href="README.en.md">English</a> | <strong>简体中文</strong></div>

# Fe DeltaSpin、LAMMPS NPT 与多 route

本例补充普通 Al 教程没有覆盖的三件事：约束磁矩标注、带自旋的 NPT、两个 route 的独立覆盖。
使用 16 原子 bcc Fe，分别从共线和倾斜磁矩初态开始；目标磁矩为教程取值，不是已验证的 Fe 磁性模型。

## 环境和资源

需要包含 DeltaSpin 扩展的 `vasp_ncl`、支持 `spin_mode 1` 的 TorchNEP、支持该模型格式的
NEPAdapters，以及含 USER-DYNSPIN/NEPAdapters 的 LAMMPS。普通 VASP 构建不能替代 DeltaSpin。
本例 `execution.targets.local` 使用当前已分配的计算资源；不要直接在集群登录节点启动训练或 DFT。
Slurm 配置方法见 [VASP 教程](../workflow-vasp-slurm/README.md)。

```bash
cd examples/workflow-vasp-deltaspin
python -m pip install -e '../..[torchnep]'
python make_candidates.py
cp vasp-resources.example.json vasp-resources.json
```

填写 `project.yaml` 中的 POTCAR 根目录，以及资源清单中 Fe/POTCAR 的真实 SHA256、TITEL 和发行版。
按所在环境设置 LAMMPS 插件；VASP 默认选择 `vasp_ncl`，自定义 launcher 可设置 `NEPTRAIN_VASP_COMMAND`。

`make_candidates.py` 生成 12 个未标注结构、`structures/fm.xyz` 和 `structures/canted.xyz`。
每原子的 `spin:R:3` 为 μB；脚本不伪造 energy/force/mforce。检查 INCAR 的泛函、SOC、k 点和磁矩定义，
并在所有标注中保持一致。框架按每帧 spin 填充 `MAGMOM` 与 `M_DELTASPIN`。

## 先标注，再启动流程

```bash
neptrain label structures/canted.xyz --backend vasp --project project.yaml --wait --output check.xyz
```

检查真实输出：

```bash
python - <<'PY'
from ase.io import read
from NepTrain.core.spin import validate_spin_dataset
frames = read('check.xyz', index=':')
print(validate_spin_dataset(frames, require_mforce=True))
print(frames[0].arrays['spin'])
print(frames[0].arrays['mforce'])
PY
```

单帧通过后，使用相同配置生成训练集：

```bash
neptrain label seed-train.xyz --backend vasp --project project.yaml --wait --output train.xyz
neptrain doctor --project project.yaml
neptrain workflow run project.yaml --prepare-only
neptrain workflow run fe-deltaspin-workflow
neptrain workflow status fe-deltaspin-workflow --jobs
```

没有配置 test，流程仍可运行。`nep.in` 明确选择 Spin NEP Lite（`spin_mode 1`）；所有后端必须支持同一模型格式。
`lammps-spin-npt.in` 使用 `dynspin/glsd/npt`。LAMMPS `units metal` 下温度为 K、压强为 bar、时间为 ps；
框架按模板填充值，不与 GPUMD 的 GPa 混用。

## 看什么结果

- `generations/0001/train/`：`training-convergence.png` 与 `training-parity-train.png`。
- `generations/0001/select/selection-pca.png`：候选与选中结构的二维分布。
- `generations/0001/label/`：真实收敛的 spin/mforce 标签与资源来源。
- `generations/0001/evaluate/`：新增标签的训练前 E/F/M 预测误差及 `acquisition-parity.png`。
- `generations/0001/update/`：更新后的训练集；两个 route 的覆盖分别记录。
- `workflow status`：说明欠缺的是精度、连续达标还是生产覆盖。

本例一代预算、10–80 步轨迹和示范阈值只用于验证接口。预期可能为 `budget_exhausted`；
这不是 DeltaSpin 失败，也不代表获得可生产使用的势函数。正式计算应扩充温度、应变、磁矩大小和方向覆盖，
独立验证能量/磁性力导数、磁矩定义及 NPT 稳定性。

本仓库的自动测试覆盖配置、结构生成和输入渲染；真实 VASP、LAMMPS 和自旋训练仍需在配置好的计算环境中验证。
