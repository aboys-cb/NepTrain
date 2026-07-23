# 独立步骤

NepTrain 的训练、MD 和 DFT 命令既可在本机运行，也可通过 `project.yaml` 中的
target 提交到 Slurm。

使用 `--project` 时，backend、模板、温度、压强和步数默认读取项目配置；命令行参数
只覆盖本次需要改变的值。

## 训练

```bash
neptrain train train.xyz \
  --backend torchnep \
  --config nep.in \
  -o nep.txt
```

加入 target 后提交 GPU 作业：

```bash
neptrain train train.xyz \
  --backend torchnep \
  --config nep.in \
  --project project.yaml \
  --target v100 \
  -o nep.txt
```

## 批量 MD

```bash
neptrain md structures/ \
  --backend lammps \
  --model nep.txt \
  --temperature 300 500 700 \
  --steps 100000 \
  --project project.yaml \
  --target cpu \
  --max-concurrent 12 \
  -o trajectories.xyz
```

Slurm target 会将“结构 × 温度”展开成带并发上限的 job array。

## 批量 DFT

```bash
neptrain dft candidates.xyz \
  --backend vasp \
  --input-file INCAR \
  --resources /shared/potpaw_PBE \
  --project project.yaml \
  --target dft \
  --structures-per-job 1 \
  --max-concurrent 20 \
  -o labeled.xyz
```

任务提交后通过统一 task 命令管理：

```bash
neptrain task status runs/dft-...
neptrain task logs runs/dft-...
neptrain task wait runs/dft-...
neptrain task retry runs/dft-...
neptrain task cancel runs/dft-...
```

只有全部 shard 成功后才会发布最终输出；已有输出默认拒绝覆盖，确认替换时使用
`--force`。
同一集群共享文件系统上的 array 会自动完成发布；跨平台任务由 `task status` 或
`task wait` 同步并发布。

平台程序默认从 `PATH` 查找。若平台要求 `srun vasp_std` 等命令，把
`NEPTRAIN_VASP_COMMAND`（或对应的 `NEPTRAIN_ABACUS_COMMAND`、
`NEPTRAIN_NEP_COMMAND`、`NEPTRAIN_GPUMD_COMMAND`）写入 target 的
`environment`。
