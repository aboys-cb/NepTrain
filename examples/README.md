<div align="center">
<a href="README.en.md">English</a> | <strong>简体中文</strong>
</div>

# NepTrain 示例入口

第一次使用时，先按标注来源选择一条路线。不要从源码目录结构猜应该运行哪个
脚本。

| 目标 | 示例 | 标注来源 | 需要的外部程序 |
|---|---|---|---|
| 用 VASP 做主动学习 | [`workflow-vasp-slurm`](workflow-vasp-slurm/README.md) | VASP 单点计算 | TorchNEP、GPUMD、VASP、Slurm |
| 用 ABACUS 做主动学习 | [`workflow-abacus-slurm`](workflow-abacus-slurm/README.md) | ABACUS 单点计算 | TorchNEP、GPUMD、ABACUS、Slurm |
| 用 DPA-3/DPA-4 蒸馏 | [`distillation-deepmd`](distillation-deepmd/README.md) | DeepMD/DPA Teacher | TorchNEP、DeePMD-kit，可选 GPUMD |
| 用 MACE 蒸馏 | [`distillation-mace`](distillation-mace/README.md) | MACE Teacher | TorchNEP、MACE，可选 GPUMD |
| 用 TACE 蒸馏 | [`distillation-tace`](distillation-tace/README.md) | TACE Teacher | TorchNEP、TACE，可选 GPUMD |

建议按下面的顺序学习：

1. 先跑示例中的独立标注命令，确认标注后端能生成 energy、forces 和 virial。
2. 再跑 Student 冒烟训练，确认标签可以被训练后端读取。
3. 最后使用示例 `project.yaml` 跑一代 workflow。
4. 把教程数据、短 MD 步数和冒烟 `nep.in` 换成自己的正式设置。

示例用于学习接口，不提供可直接用于生产的势函数。VASP/ABACUS 的种子结构现在不带标签，
必须使用同一 DFT 后端和同一输入设置标注后再训练；不混合 EMT 与 DFT 标签。

## 选择有代表性的案例

| 案例/配置 | 体系 | 采样 | 主要学习内容 |
|---|---|---|---|
| VASP `project.yaml` | 周期 Al | GPUMD NPT | Slurm、多执行 target、固定 POTCAR、同源种子标注 |
| VASP `project-lammps.yaml` | 周期 Al | LAMMPS NPT | 相同标注与训练设置下切换 MD 后端；压强用 bar |
| ABACUS `project.yaml` | 周期 Al | GPUMD NPT | 平面波、UPF 来源与后端替换 |
| MACE / TACE | 周期 Al | GPUMD NPT | 不同 Teacher 的调用、模型哈希与标签来源 |
| DeepMD / DPA | 真空盒单水分子 | GPUMD NVT | H/O 多元素、模型 head、固定盒分子标注 |
| [VASP DeltaSpin](workflow-vasp-deltaspin/README.md) | 周期 Fe，自旋 | LAMMPS NPT，两条 route | 三分量磁矩、mforce、共线/倾斜初态独立覆盖 |

所有案例配置均面向新版主动学习流程，独立 test 可省略。GPUMD 压强单位为 GPa；LAMMPS
`units metal` 为 bar，配置不自动跨后端换算。水分子真空盒保持 NVT，不对真空体积做 NPT。
短轨迹只检验接口，不代表 NPT/NVT 已平衡。温度、时间步长、耦合参数和收敛阈值必须按实际问题验证。

当前案例还不覆盖复杂合金缺陷、液体相变或大规模磁性训练的科学验收，不能从短教程推断这些能力已经验证。
本地检查覆盖所有 YAML、种子生成、模板替换和阶段契约；真实后端需用户在配好资源的环境中按教程运行。
