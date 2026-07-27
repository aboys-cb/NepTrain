# 安装

NepTrain 要求 Python 3.10 或更高版本：

```bash
pip install NepTrain
```

TorchNEP：

```bash
pip install torch
pip install 'NepTrain[torchnep]'
```

MACE Teacher 蒸馏标注：

```bash
pip install 'NepTrain[mace]'
```

该 extra 提供 NepTrain 内部 MACE 标注适配器所需的运行时。请先按机器 CUDA
版本安装 PyTorch，再安装此 extra。

DeepMD / DPA Teacher 蒸馏标注：

```bash
pip install 'NepTrain[deepmd]'
```

该 extra 安装 NepTrain 内部 DeepMD 标注适配器所需的稳定版 DeePMD-kit
PyTorch 后端。它支持 DPA-3 和其它稳定版 DeepMD 模型。DPA-4
从 DeePMD-kit 3.2 开始提供；在 3.2 仍为预发布版时需显式安装：

```bash
pip install --pre 'deepmd-kit[torch]>=3.2.0b0,<4'
pip install NepTrain
```

手动采样不提供 NEP 模型、需要 SOAP 描述符时：

```bash
pip install 'NepTrain[soap]'
```

LAMMPS、VASP、ABACUS 和赝势由用户或计算平台提供。推荐把 module、PATH 和
`LAMMPS_PLUGIN_PATH` 写入 execution target 的 `setup_script`，然后运行：

```bash
neptrain doctor --project project.yaml
```

`doctor` 会从项目读取 training、MD、labeling backend、stage target、
`setup_script` 和 `environment`，分别在实际目标环境检查 `nep`/TorchNEP、
GPUMD/LAMMPS、VASP/ABACUS 或 MACE/DeepMD；通常不需要再手写 backend 参数。

NepTrain 不再读取 `~/.NepTrain` 中的旧环境配置。所有可复现执行环境都应进入
schema-v8 `project.yaml` 的 target 或对应环境脚本。
