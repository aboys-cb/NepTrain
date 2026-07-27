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

建议按下面的顺序学习：

1. 先跑示例中的独立标注命令，确认标注后端能生成 energy、forces 和 virial。
2. 再跑 Student 冒烟训练，确认标签可以被训练后端读取。
3. 最后使用示例 `project.yaml` 跑一代 workflow。
4. 把教程数据、短 MD 步数和冒烟 `nep.in` 换成自己的正式设置。

示例的任务是验证软件链路，不是提供可直接发表或生产模拟的势函数。VASP 和
ABACUS 示例附带的初始数据由 ASE EMT 生成，只用于让新用户跑通工作流机械过程；
正式项目必须换成与目标理论水平一致的第一性原理数据。
