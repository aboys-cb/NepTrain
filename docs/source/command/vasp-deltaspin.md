# VASP DeltaSpin

使用带 DeltaSpin 扩展的 VASP 6 `vasp_ncl`，将每个输入结构的 `spin:R:3` 作为目标磁矩，输出 energy、forces、virial、实际收敛磁矩 `spin` 和磁性力 `mforce`。普通 VASP 的 `ISPIN=2` 仍只产生普通能量/力标签。

## 创建流程

```bash
neptrain workflow init --profile slurm --spin --dft-backend vasp --directory fe-spin
cd fe-spin
```

命令生成自旋 LAMMPS 模板、DeltaSpin INCAR 和 POTCAR manifest 模板。准备带 `spin/mforce` 的训练集、NEP 训练输入和带 `spin` 的采样初态，再填写资源路径与 Slurm 分区。每个原子都需要三分量 `spin`，非磁性原子的零目标也要显式写出。

默认启动命令使用 `mpirun -n N vasp_ncl`；普通 VASP 标注仍使用 `vasp_std`。若集群使用其它启动方式，在 labeling target 的 `environment` 或 setup script 设置 `NEPTRAIN_VASP_COMMAND`，例如 `srun vasp_ncl`。该程序必须是支持 `LDELTASPIN` 的构建，仅有同名可执行文件还不够。

```yaml
md:
  backend: lammps
  spin: true
labeling:
  backend: vasp
  input_path: ./INCAR
  resource_path: /path/to/potpaw_PBE
  potcar_manifest_path: ./vasp-resources.json
  kpoint_mode: auto
```

这是生成项目中的相关配置片段。POTCAR 的路径、TITEL、SHA256 和 family 仍按普通 VASP 流程校验。训练、MD 和标注节点需要使用包含本功能的同一版本 NepTrain。

## INCAR 怎么写

以下是内置完整模板。数值设置参考 Fe2 示例；混合参数、ENCUT、展宽和 k 点密度需要针对材料验证。模板默认保留 SOC，可按物理问题将 `LSORBIT` 改为 `.FALSE.`。

```{literalinclude} ../../../src/NepTrain/core/dft/vasp/INCAR.deltaspin
:language: text
```

`LDELTASPIN=.TRUE.` 开启约束，`LNONCOLLINEAR=.TRUE.` 开启三分量磁矩。标注保持固定结构，要求 `IBRION=-1`、`NSW=0`；当前接口使用 `SAXIS=0 0 1`，输入和输出的自旋坐标因此与结构笛卡尔坐标一致。

`M_DELTASPIN` 由每帧的 `spin` 自动生成；模板中已有的值也会被替换。`DELTASPIN_ATOMS` 自动生成为每原子一个 1，`DELTASPIN_COMPONENTS` 为 `1 1 1`。不要在通用模板中写死原子数或某一帧的目标磁矩。部分原子约束、部分分量约束和非零 `DELTASPIN_CONSTRAINT_MODE` 会被拒绝。

`MAGMOM` 是初始猜测，流程统一将其设为每帧输入 `spin`，与 `M_DELTASPIN` 使用相同的目标磁矩；模板中已有的 `MAGMOM` 也会被覆盖。NepTrain 会随 POSCAR 一起重排这两个字段，输出再还原原子顺序。非共线计算写入 `ISPIN=1`，避免 ASE 自动生成共线设置。

`DELTASPIN_MOMENT_DEF` 支持 0（PROCAR 投影磁矩）、1（PAW-POU）、2（PAW onsite all-l）；输入自旋与训练数据必须采用同一磁矩定义，不应仅因字段形状相同就混合。模板使用 0，并设置 `LORBIT=11`。

`DELTASPIN_TOL=1E-6` 是磁矩分量误差容限，单位 μB；`EDIFF=1E-7` 是电子能量收敛阈值，单位 eV。`EDIFF_RHO=0` 关闭这个扩展的额外密度残差门槛。`DELTASPIN_ACTIVATION_RATIO=10` 保留参考示例设置；这些数值不是对所有材料的收敛保证。

## 先标注一个结构

从已有 POSCAR 构造 Fe2 的自旋输入，两个目标分别为 `(0.35, 0, 2.677)` 和 `(-0.35, 0, 2.677)` μB：

```python
import numpy as np
from ase.io import read, write
atoms = read("POSCAR")
assert atoms.get_chemical_symbols() == ["Fe", "Fe"]
atoms.set_array("spin", np.array([[0.35, 0, 2.677], [-0.35, 0, 2.677]]))
write("selected-spin.xyz", atoms, format="extxyz")
```

```bash
neptrain label selected-spin.xyz --backend vasp --project project.yaml --wait --output labeled-spin.xyz
```

运行前完成项目的资源和执行 target 配置。`label` 与 workflow 使用同一标注接口；单结构通过后再启动迭代。仅有普通 POSCAR 或 `MAGMOM` 不足以定义流程中的目标，必须提供规范的 `spin:R:3` 输入。

## 检查结果

OUTCAR 必须包含一次 `DeltaSpin final status`，其中 `SCF convergence` 和 `Moment constraint` 都为 `reached`，并满足报告的误差容限。缺失、非有限、部分约束或原子顺序不匹配的磁矩/磁性力表均拒收，输入中残留的旧 `mforce` 不能代替新结果。

输出 `spin` 来自实际收敛磁矩表，单位 μB；`mforce` 来自 `Magnetic force (eV/uB)` 表，按 `spin_lambda` 原符号保存，对应 `L=E+lambda.(M-M_target)`。能量、原子力和 virial 沿用 ASE 的 VASP 读取路径。`vasp-result.json` 保存输入/输出哈希、磁矩定义、误差和单位。

```python
from ase.io import read
frame = read("labeled-spin.xyz")
print(frame.arrays["spin"])
print(frame.arrays["mforce"])
print(frame.info["deltaspin_max_error"], frame.info["deltaspin_tolerance"])
```

接口面向输出 `ion elem cx cy cz Mx My Mz |M|` 和 `MFx MFy MFz` 表的 VASP 6 DeltaSpin；不读取旧 VASP65 的磁性力格式。不同版本或磁矩定义的数据混用前，需要独立核对符号、单位和导数一致性。
