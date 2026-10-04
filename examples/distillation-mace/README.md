<div align="center">
<a href="README.en.md">English</a> | <strong>简体中文</strong>
</div>

# 用 MACE Teacher 蒸馏一个 NEP Student

这个教程先用固定的 MACE-MP-0 checkpoint 标注12 个带局域位移的 Al 晶体结构，再训练
TorchNEP Student，最后可选跑一代采样流程：

```text
候选结构 → MACE Teacher → energy/forces/virial
         → TorchNEP Student → GPUMD 采样 → 再标注、训练前评估与更新数据
```

## 1. 进入示例并创建环境

```bash
cd examples/distillation-mace
python -m pip install -e '../..[mace,torchnep]'
```

从当前源码安装，确保 NepTrain 与附带的占位符模板匹配；MD 计算节点也应使用同一
源码版本。`gpumd-npt.in` 由 NepTrain 渲染后运行，不要直接交给 GPUMD。

先按机器驱动安装合适的 PyTorch，再安装上面的 extra。确认环境：

```bash
command -v neptrain
python -c 'import torch, mace; print(torch.__version__, torch.cuda.is_available())'
```

独立标注可以先用 CPU；`project.yaml` 的完整 workflow 使用 CUDA 和 GPUMD。

## 2. 下载固定 Teacher

```bash
python download_model.py
```

脚本下载 MACE-MP-0 128-channel small checkpoint，保存为
`mace-mp-0-small.model`，并校验脚本中固定的 SHA256。文件已存在时也会重新
校验，避免无声使用不同权重。

本例不用 `mace_mp()` 的动态默认别名，因为 MACE 版本升级可能改变默认模型。
NepTrain 将本地 checkpoint 的真实 SHA256 写入标签 provenance。

## 3. 生成候选结构

```bash
python make_candidates.py
```

`candidates.xyz` 包含12 个不同晶格缩放与局域位移的 fcc Al 结构：

```bash
python - <<'PY'
from ase.io import read
frames = read("candidates.xyz", index=":")
print("frames:", len(frames))
print("atoms per frame:", len(frames[0]))
print("volumes:", [round(frame.get_volume(), 3) for frame in frames])
PY
```

## 4. 用 MACE 标注

先用 CPU 跑通接口：

```bash
neptrain label candidates.xyz \
  --backend model \
  --model mace-mp-0-small.model \
  --model-name mace-mp-0-small \
  --runner 'neptrain model-worker mace' \
  --device cpu \
  --precision float32 \
  --structures-per-job 2 \
  --wait \
  --output labeled.xyz
```

有 CUDA 时可把 `--device cpu` 改为 `--device cuda`。检查标签：

```bash
python inspect_labels.py
```

最后一行应以 `OK:` 开头。12 个 frame 必须共享一个 Teacher SHA256，并同时具有
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
  --seed 20260727 \
  --workdir student-train \
  --output student-nep.txt
```

`nep-smoke.in` 只训练 10 个 epoch。它证明 Teacher 标签能被 TorchNEP 读取并
发布 `nep.txt`，但生成的模型没有科学使用价值。

```bash
test -s student-nep.txt
test -s student-train/jobs/000000/calculation/training-report.json
test -s student-train/jobs/000000/calculation/training-convergence.png
```

训练图只生成 PNG。

## 6. 跑一代采样流程

确认 GPUMD 和项目环境：

```bash
command -v gpumd
neptrain doctor --project project.yaml
```

`project.yaml` 使用本地 process target，适合在已经分配 GPU 的节点运行。
`nep-workflow.in` 使用 500 个 epoch，`gpumd-npt.in` 做极短的 NPT 采样。

```bash
neptrain workflow run project.yaml --prepare-only
neptrain workflow run mace-distillation-workflow
neptrain workflow status mace-distillation-workflow --jobs
```

准备完成后，若希望在当前终端观察 controller，用以下命令替代上面的后台启动命令（两种启动方式选一种）：

```bash
neptrain workflow run mace-distillation-workflow --foreground
```

## 7. 跑完检查什么

| 位置 | 内容 |
|---|---|
| `generations/0001/train/` | 初始 Student、`training-convergence.png` 和 `training-parity-train.png` |
| `generations/0001/explore/` | GPUMD 轨迹和健康报告 |
| `generations/0001/select/` | FPS 结果和 `selection-pca.png` |
| `generations/0001/label/selected-labels.xyz` | MACE 新标签 |
| `generations/0001/label/label-provenance.json` | runner、模型名和 SHA256 |
| `generations/0001/update/` | 合并后的训练集 |
| `generations/0001/evaluate/` | 新标签的训练前预测误差和 `acquisition-parity.png` |

本例 `max_model_generations: 1`，所以末尾 `budget_exhausted` 表示教程预算用完，
不代表任务失败。

## 8. 正式使用前

- 根据目标体系选择与元素范围、理论水平和许可证匹配的 MACE checkpoint。
- 使用覆盖目标温压和结构空间的候选集，不要只用本例的 12 个微扰晶胞。
- 按目标体系设置 `workflow.convergence`，以新增结构的训练前误差判断收敛；独立测试集可选，仅供辅助诊断。
- 扩大 Student 网络、训练 epoch 和 workflow 代数。
- 把本地 target 换成适合集群的 Slurm target。
- 不要混合来自不同能量基准或理论水平的 Teacher 标签。

## 常见问题

| 现象 | 常见原因 | 处理 |
|---|---|---|
| `model SHA256 mismatch` | 下载文件变化或不完整 | 删除 `.part`/模型文件后重新下载 |
| `MACE is not installed` | extra 装在另一个环境 | 检查 `which python` 和 `pip show mace-torch` |
| CUDA OOM | checkpoint 或 shard 太大 | 降低 `structures_per_job`，先用 CPU 冒烟 |
| `returned non-finite labels` | Teacher 不适合该结构或输入异常 | 独立用 MACE ASE calculator 检查 |
| 输入含 `spin` 被拒绝 | MACE runner 不生成 `mforce` | 使用支持磁力标签的 Teacher |

参考：[MACE Foundation Models](https://mace-docs.readthedocs.io/en/latest/guide/foundation_models.html)。

## 流程版本与预期结果

新建流程统一使用 `active_learning_v4`：
`train → explore → select → label → evaluate → update`。
`train` 同时检查模型来源、启用模型并可选输出测试误差；`evaluate` 用新标签评估采样模型的训练前预测。
`project.yaml` 中的阈值仅演示配置，不代表模型已经达到该精度。一代采样预算通常以
`budget_exhausted` 结束，属于预期结果，不是后端失败。更新的数据由下一代训练；只有精度和
生产覆盖均满足条件才进入最终训练。
NPT 使用 0 GPa 的各向同性 `npt_scr`，粗略弹性模量为 100 GPa，温压耦合参数为 100/1000 步；小晶胞和短轨迹只验证接口，不代表平衡采样。
