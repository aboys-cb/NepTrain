# 教程与示例

第一次使用 NepTrain 时，先按标注来源选择示例：

| 标注来源 | 教程 | 适合先验证什么 |
|---|---|---|
| VASP | [VASP + Slurm workflow](https://github.com/aboys-cb/NepTrain/tree/master/examples/workflow-vasp-slurm) | POTCAR manifest、Slurm target、真实 VASP 标注 |
| VASP DeltaSpin | [Fe 自旋 workflow](https://github.com/aboys-cb/NepTrain/tree/master/examples/workflow-vasp-deltaspin) | 目标磁矩、磁力标签、LAMMPS spin NPT 与多 route |
| ABACUS | [ABACUS + Slurm workflow](https://github.com/aboys-cb/NepTrain/tree/master/examples/workflow-abacus-slurm) | UPF/ORB manifest、Slurm target、真实 ABACUS 标注 |
| DPA-3/DPA-4 | [DeepMD 蒸馏](https://github.com/aboys-cb/NepTrain/tree/master/examples/distillation-deepmd) | 公开 DPA-3 下载、模型标注、Student 和完整 workflow |
| MACE | [MACE 蒸馏](https://github.com/aboys-cb/NepTrain/tree/master/examples/distillation-mace) | 固定 checkpoint、模型标注、Student 和完整 workflow |
| TACE | [TACE 蒸馏](https://github.com/aboys-cb/NepTrain/tree/master/examples/distillation-tace) | 固定 foundation model、预测字段归一化、Student 和完整 workflow |

普通 VASP 示例还提供 `project-lammps.yaml`，用于对照 GPUMD 与 LAMMPS NPT。
所有案例配置均为 `schema_version: 8`。

每个教程都按相同顺序组织：

1. 安装并检查命令；
2. 准备结构、标签和外部资源；
3. 先跑一次独立标注；
4. 检查 energy、forces、virial 和 provenance；
5. 准备并启动一代采样流程；
6. 查看 stage、日志、PNG 图和最终模型状态；
7. 将教程参数替换为正式项目参数。

VASP/ABACUS 示例先生成无标签结构，再使用与后续 workflow 一致的后端和输入
标注初始训练集，不生成或混入 EMT 标签。块体示例使用 NPT；DeepMD 的单水分子
示例保持固定盒 NVT。所有示例的小数据、短训练和短轨迹都只用于检查接口与流程，
不代表势函数精度、平衡或生产可用性已经通过验证。
