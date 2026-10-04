<div align="center">
<a href="README.en.md">English</a> | <strong>简体中文</strong>
</div>

# 用 DPA-3 Teacher 蒸馏一个 NEP Student

这个教程从一个公开 DPA-3 模型开始，先完成独立标注和 Student 冒烟训练，再跑一
代完整 workflow：

```text
候选结构 → DPA-3 Teacher → energy/forces/virial
         → TorchNEP Student → GPUMD 采样 → 再标注、训练前评估与更新数据
```

主教程使用 DeePMD-kit 官方内置的 `DPA-3.2-5M` 和 `OMol25` head。DPA-4
使用同一个 NepTrain runner，但模型准备方式不同，见最后一节。

## 1. 进入示例并创建独立环境

从仓库根目录执行：

```bash
cd examples/distillation-deepmd
python -m pip install -e '../..[deepmd,torchnep]'
```

从当前源码安装，确保 NepTrain 与附带的占位符模板匹配；MD 计算节点也应使用同一
源码版本。`gpumd-nvt.in` 由 NepTrain 渲染后运行，不要直接交给 GPUMD。

确认命令可见：

```bash
command -v neptrain
command -v dp
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

独立标注可以使用 CPU；本例 `project.yaml` 的训练、Teacher 标注和 GPUMD
workflow 使用 CUDA。如果 `torch.cuda.is_available()` 是 `False`，先安装与机器
驱动匹配的 PyTorch/CUDA 环境。

## 2. 下载并检查 Teacher

```bash
python download_model.py
dp --pt show DPA-3.2-5M.pt model-branch type-map
```

脚本使用 `dp pretrained download` 下载模型，将固定文件复制为
`DPA-3.2-5M.pt`，并打印 SHA256。`dp --pt show` 的输出中应包含 `OMol25`
branch 和 H/O 元素。

NepTrain 接收本地模型文件，而不是会随安装版本变化的别名。标注任务会记录该文件
的 SHA256。

## 3. 生成三个候选结构

```bash
python make_candidates.py
```

这会写出 `candidates.xyz`，其中是三个轻微缩放 O–H 键长的周期水分子结构：

```bash
python - <<'PY'
from ase.io import read
frames = read("candidates.xyz", index=":")
print("frames:", len(frames))
print("symbols:", frames[0].get_chemical_formula())
print("cell volume:", frames[0].get_volume())
PY
```

Teacher runner 需要正体积晶胞，因为它必须把 ASE stress 转成 virial。

## 4. 用 DPA-3 标注

```bash
neptrain label candidates.xyz \
  --backend model \
  --model DPA-3.2-5M.pt \
  --model-name dpa-3.2-5m-omol25 \
  --runner 'neptrain model-worker deepmd --head OMol25' \
  --device cuda \
  --precision float32 \
  --structures-per-job 2 \
  --wait \
  --output labeled.xyz
```

`--head OMol25` 属于 DeepMD runner，所以必须放在带引号的 runner 字符串内。
NepTrain 会在后面追加统一的 `--model`、`--input`、`--output`、`--device` 和
`--precision` 参数。

检查输出：

```bash
python inspect_labels.py
```

最后一行应以 `OK:` 开头，并显示三个 frame 共用一个 Teacher SHA256。每个 frame
必须同时具有有限的 energy、forces 和形状为 `(3, 3)` 的 virial。

## 5. 训练一个 Student 冒烟模型

```bash
neptrain train labeled.xyz \
  --backend torchnep \
  --config nep-smoke.in \
  --device cuda \
  --torch-backend bmm \
  --precision float32 \
  --no-compile \
  --seed 20260727 \
  --workdir student-train \
  --output student-nep.txt
```

`nep-smoke.in` 只有 10 个 epoch，用来确认 DeepMD 标签能进入 TorchNEP 并发布
`nep.txt`。这个模型没有科学使用价值，也不应用于正式 MD。

跑完检查：

```bash
test -s student-nep.txt
test -s student-train/training-report.json
test -s student-train/training-convergence.png
```

训练图只生成 PNG。

## 6. 跑一代完整 workflow

完整链路还需要 `gpumd`：

```bash
command -v gpumd
neptrain doctor --project project.yaml
```

本例使用 `gpumd-nvt.in`：固定真空盒，50 K Nosé–Hoover 恒温，不使用压强耦合。0.1 fs 步长和 2–16 步只验证接口，不能证明达到热平衡。`nep-workflow.in` 使用 500 个 epoch。

先准备，再启动：

```bash
neptrain workflow run project.yaml --prepare-only
neptrain workflow run deepmd-distillation-workflow
neptrain workflow status deepmd-distillation-workflow --jobs
```

如果要在当前终端观察 controller，可将最后两步替换为：

```bash
neptrain workflow run project.yaml --foreground
```

## 7. 跑完检查什么

| 位置 | 内容 |
|---|---|
| `generations/0001/train/` | 初始 Student 和训练曲线 PNG |
| `generations/0001/explore/` | GPUMD 轨迹与 `trajectory-health.json` |
| `generations/0001/select/` | FPS 选中的候选 |
| `generations/0001/label/selected-labels.xyz` | DPA-3 新标签 |
| `generations/0001/label/label-provenance.json` | runner、head、模型名和 SHA256 |
| `generations/0001/update/` | 合并后的 Student 训练集 |
| `generations/0001/evaluate/` | 激活结果与评估产物 |

示例只允许一代。最后显示 `budget_exhausted` 表示一代教程预算用完，不是
workflow 失败。查看 `status --jobs` 时，八个 stage 应进入完成状态；如果
Teacher 或 MD 失败，对应 stage 会明确显示失败。

## 8. 换成 DPA-4

DPA-4/SeZM 是 DeePMD-kit 的 PyTorch-only 模型，冻结后使用 `.pt2`。先安装支持
DPA-4 的 DeePMD-kit 版本，并按官方 DPA-4 文档在目标 GPU 上 freeze/export：

```bash
pip install --pre 'deepmd-kit[torch]>=3.2.0b0,<4'
dp --pt freeze -c model.ckpt.pt -o frozen-dpa4
```

然后仍使用同一个 runner：

```bash
neptrain label candidates.xyz \
  --backend model \
  --model frozen-dpa4.pt2 \
  --model-name dpa4-sezm \
  --runner 'neptrain model-worker deepmd' \
  --device cuda \
  --precision float32 \
  --wait \
  --output labeled-dpa4.xyz
```

多任务模型在 runner 中加入 `--head <name>`。`.pt2` 会包含与 freeze 环境相关的
编译产物，所以应先在实际部署 GPU 上做独立标注冒烟，再写进 workflow。本仓库
已实际跑通 DPA-3 workflow；DPA-4 部分是统一接口说明，不冒充公开 DPA-4
checkpoint 的端到端验证。

## 常见问题

| 现象 | 常见原因 | 处理 |
|---|---|---|
| `dp: command not found` | DeePMD extra 没装进当前环境 | 重新安装并确认 conda 环境 |
| `head ... not found` | 模型没有该 branch | 用 `dp --pt show ... model-branch` 查看 |
| `cannot access a CUDA device` | 当前进程没拿到 GPU | 检查 Slurm GPU 资源和 PyTorch CUDA |
| `returned non-finite labels` | Teacher 推理结果异常 | 先用同一模型/结构运行 `dp test` 或 ASE |
| GPUMD 输出 NaN | Student 或 MD 参数不稳定 | 看健康报告；本例不要把 NVT 改成强耦合 NVT |
| 输入含 `spin` 被拒绝 | runner 不生成 `mforce` | 使用真正支持磁力标签的 Teacher backend |

参考：[DeePMD-kit 内置模型下载](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/pretrained.html)、
[官方 DPA-3 蒸馏教程](https://docs.deepmodeling.com/projects/deepmd/en/latest/getting-started/dpa3_cyclohexane_distillation.html)、
[DPA-4 文档](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dpa4.html)。

## 流程版本与预期结果

新建流程统一使用 `active_learning_v4`：
`train → explore → select → label → evaluate → update`。
`train` 同时检查模型来源、启用模型并可选输出测试误差；`evaluate` 用新标签评估采样模型的训练前预测。
`project.yaml` 中的阈值仅演示配置，不代表模型已经达到该精度。一代采样预算通常以
`budget_exhausted` 结束，属于预期结果，不是后端失败。更新的数据由下一代训练；只有精度和
生产覆盖均满足条件才进入最终训练。
