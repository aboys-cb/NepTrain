<div align="center">
<a href="README.en.md">English</a> | <strong>简体中文</strong>
</div>

# 用 TACE Teacher 蒸馏一个 NEP Student

这个教程用固定版本的 `TACE-OAM-7M` 标注三个 Al 晶体结构，再训练
TorchNEP Student，最后可选跑一代完整 workflow：

```text
候选结构 → TACE Teacher → energy/forces/virial
         → TorchNEP Student → GPUMD 采样 → 再标注与重训
```

## 1. 进入示例并安装环境

TACE 目前按官方仓库安装。本教程固定到已经核对过 `tace-eval` 接口的 commit：

```bash
cd examples/distillation-tace
pip install torch
pip install 'NepTrain[torchnep]'
pip install \
  'TACE[cueq12] @ git+https://github.com/xvzemin/tace.git@4b977dcc13ee87d8ba6cceba3ffb7abe43c087c8'
export TACE_USE_CUE=1
```

先确认命令来自同一个环境：

```bash
command -v neptrain
command -v tace-eval
python -c 'import torch, tace; print(torch.__version__, torch.cuda.is_available())'
```

独立标注可以先用 CPU；`project.yaml` 的完整 workflow 使用 CUDA 和 GPUMD。
`cueq12` 安装 cuEquivariance 的 CUDA 12 算子。官方 wheel 需要 Ampere
或更新的 GPU；Sai V100 实测会报 `cudaErrorNoKernelImageForDevice`。V100 上请安装
不带 `cueq12` 的 TACE 并保持 `TACE_USE_CUE=0`。

## 2. 下载固定 Teacher

```bash
python download_model.py
```

脚本从 TACE 官方 Hugging Face 仓库的固定 revision 下载
`TACE-OAM-7M.pt`，并校验固定 SHA256。文件已存在时也会重新校验。
NepTrain 还会把本地模型的真实 SHA256 写入标签 provenance。

## 3. 生成候选结构

```bash
python make_candidates.py
```

`candidates.xyz` 包含三个不同晶格缩放的 fcc Al 结构：

```bash
python - <<'PY'
from ase.io import read
frames = read("candidates.xyz", index=":")
print("frames:", len(frames))
print("atoms per frame:", len(frames[0]))
print("volumes:", [round(frame.get_volume(), 3) for frame in frames])
PY
```

## 4. 用 TACE 标注

先用 CPU 跑通接口：

```bash
neptrain label candidates.xyz \
  --backend model \
  --model TACE-OAM-7M.pt \
  --model-name TACE-OAM-7M \
  --runner 'neptrain model-worker tace --fidelity-index 0' \
  --device cpu \
  --precision float32 \
  --structures-per-job 2 \
  --wait \
  --output labeled.xyz
```

TACE 官方 `tace-eval` 先在临时目录写预测 extxyz。NepTrain 随后检查 frame 数量、
有限值和标签形状，并把 `TACE_energy`、`TACE_forces` 以及
`TACE_stress`/`TACE_virials` 归一成标准 `energy`、`forces`、`virial` 后再发布。

检查标签：

```bash
python inspect_labels.py
```

最后一行应以 `OK:` 开头。三个 frame 必须共享一个 Teacher SHA256，并同时具有
有限的 energy、forces 和 `(3, 3)` virial。

## 5. 训练 Student 冒烟模型

```bash
neptrain train labeled.xyz \
  --backend torchnep \
  --config nep-smoke.in \
  --device cuda \
  --torch-backend bmm \
  --precision float32 \
  --no-compile \
  --seed 20260728 \
  --workdir student-train \
  --output student-nep.txt
```

`nep-smoke.in` 只训练 10 个 epoch，用来确认 TACE 标签能被 TorchNEP 读取。
生成的模型没有科学使用价值。

```bash
test -s student-nep.txt
test -s student-train/training-report.json
test -s student-train/training-convergence.png
```

## 6. 跑一代完整 workflow

```bash
command -v gpumd
neptrain doctor --project project.yaml
neptrain workflow run project.yaml --prepare-only
neptrain workflow run tace-distillation-workflow
neptrain workflow status tace-distillation-workflow --jobs
```

也可以在前台观察 controller：

```bash
neptrain workflow run project.yaml --foreground
```

本例只允许一代。末尾 `budget_exhausted` 表示教程预算用完，不代表失败。

## 7. 跑完检查什么

| 位置 | 内容 |
|---|---|
| `generations/0001/train/` | 初始 Student 和训练曲线 PNG |
| `generations/0001/md/` | GPUMD 轨迹和健康报告 |
| `generations/0001/select/` | FPS 选择结果 |
| `generations/0001/label/selected-labels.xyz` | TACE 新标签 |
| `generations/0001/label/label-provenance.json` | runner、模型名和 SHA256 |
| `generations/0001/dataset/` | 合并后的训练集 |
| `generations/0001/retrain/` | 重训 Student |
| `generations/0001/evaluate/` | 激活结果 |

## 8. Spin Teacher 边界

TACE 支持非共线磁力模型，但普通 `TACE-OAM-7M` 不等于磁性 Teacher。只有当
checkpoint 真实输出 `noncollinear_magnetic_forces` 时，NepTrain 才允许标注含
`spin` 的结构，并将其映射为标准 `mforce`。缺少磁力输出会直接失败，不会补零。

## 正式使用前

- 根据目标元素、理论水平和许可证选择或微调 TACE checkpoint。
- 不要混合不同 fidelity 或不同能量基准的 Teacher 标签。
- 使用独立验证集，并设置有物理意义的验收阈值。
- 增大 Student、训练 epoch、候选空间和 workflow 代数。
- 在目标 GPU 上先跑独立标注，再切换为 Slurm target。

## 常见问题

| 现象 | 常见原因 | 处理 |
|---|---|---|
| `tace-eval is not installed` | TACE 装在另一个环境 | 检查 `command -v tace-eval` |
| `model SHA256 mismatch` | 下载不完整或模型文件变化 | 删除 `.part` 和模型后重下 |
| `returned neither virial nor stress` | checkpoint 没有应力/virial head | 换用包含训练所需标签的模型 |
| CUDA OOM | 模型或 shard 太大 | 降低 `structures_per_job`，先用 CPU |
| spin 输入缺少 magnetic forces | 普通 checkpoint 不输出 `mforce` | 使用真正训练了非共线磁力的 TACE 模型 |

参考：[TACE 官方仓库](https://github.com/xvzemin/tace)、
[TACE 推理命令](https://tace.readthedocs.io/en/latest/guide/scripts.html)。
