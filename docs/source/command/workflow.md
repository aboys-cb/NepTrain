# 自动迭代

## 创建项目

```bash
neptrain workflow init --profile slurm --directory fe-project
cd fe-project
neptrain doctor --project project.yaml
```

项目只接受 `schema_version: 4`。温度、压强和 MD 步数统一位于 `md`，不再保留
旧字段或运行期迁移。

## 准备和运行

```bash
neptrain workflow run project.yaml --prepare-only
neptrain workflow run project.yaml
```

输出目录默认使用 `workflow.id`。Controller 默认脱离终端，按 ledger 推进：

```text
train → explore → select → label → diagnose → merge → retrain → evaluate
```

训练、MD 和 DFT stage 使用与独立命令相同的 Adapter 和 execution target。

## 状态与恢复

```bash
neptrain workflow status fe-workflow
neptrain workflow status fe-workflow --jobs
neptrain workflow resume fe-workflow
neptrain workflow stop fe-workflow
neptrain workflow extend fe-workflow 5
```

Controller 不依赖 Slurm `afterok`。失败后只重跑 ledger 中未完成的阶段；已完成
workflow 重复运行是安全 no-op。

默认停止只退出 Controller，保留当前计算任务：

```bash
neptrain workflow stop fe-workflow
```

确认整个流程已经作废时，可同时取消当前 process 或 Slurm 作业：

```bash
neptrain workflow stop fe-workflow --cancel-jobs
```

取消动作会记录到 workflow 历史；后续恢复会创建新的 stage attempt。
