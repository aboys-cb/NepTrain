# 教程与示例

第一次使用 NepTrain 时，先按标注来源选择示例：

| 标注来源 | 教程 | 适合先验证什么 |
|---|---|---|
| VASP | [VASP + Slurm workflow](https://github.com/aboys-cb/NepTrain/tree/master/examples/workflow-vasp-slurm) | POTCAR manifest、Slurm target、真实 VASP 标注 |
| ABACUS | [ABACUS + Slurm workflow](https://github.com/aboys-cb/NepTrain/tree/master/examples/workflow-abacus-slurm) | UPF/ORB manifest、Slurm target、真实 ABACUS 标注 |
| DPA-3/DPA-4 | [DeepMD 蒸馏](https://github.com/aboys-cb/NepTrain/tree/master/examples/distillation-deepmd) | 公开 DPA-3 下载、模型标注、Student 和完整 workflow |
| MACE | [MACE 蒸馏](https://github.com/aboys-cb/NepTrain/tree/master/examples/distillation-mace) | 固定 checkpoint、模型标注、Student 和完整 workflow |
| TACE | [TACE 蒸馏](https://github.com/aboys-cb/NepTrain/tree/master/examples/distillation-tace) | 固定 foundation model、预测字段归一化、Student 和完整 workflow |

每个教程都按相同顺序组织：

1. 安装并检查命令；
2. 准备结构、标签和外部资源；
3. 先跑一次独立标注；
4. 检查 energy、forces、virial 和 provenance；
5. 准备并启动一代 workflow；
6. 查看 stage、日志、PNG 图和最终模型状态；
7. 将教程参数替换为正式项目参数。

VASP/ABACUS 示例的初始数据使用 ASE EMT，只用于跑通工作流机械过程。它不能与
正式第一性原理标签混合作为生产训练集。蒸馏示例的小数据和短训练同样只用于验证
接口。
