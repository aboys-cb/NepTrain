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

LAMMPS、VASP、ABACUS 和赝势由用户或计算平台提供。推荐把 module、PATH 和
`LAMMPS_PLUGIN_PATH` 写入 execution target 的 `setup_script`，然后运行：

```bash
neptrain doctor --project project.yaml
```

NepTrain 不再读取 `~/.NepTrain` 中的旧环境配置。所有可复现执行环境都应进入
schema-v4 `project.yaml` 的 target 或对应环境脚本。
