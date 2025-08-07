# train
**Description:**  
Performs automatic training for NEP.
## Input Parameters

:::{important}
- You should use `NepTrain init <type>` to generate `job.yaml`, and then use `NepTrain train job.yaml` to start the training task.

- After the program runs, a `restart.yaml` file will be generated. To continue training, you can use `NepTrain train restart.yaml`.
::: 
**Usage:**  
```bash
NepTrain train <config_path>
```

**Options:**
- `<config_path>`
  Path to configuration file, such as `job.yaml`.
## Output
During training, intermediate data and the resulting potential are
written to the working directory. A `restart.yaml` file is created after
each iteration, allowing you to resume the workflow with
`NepTrain train restart.yaml`.

## Example

### 初始化操作 / Initialization
在命令行输入 `NepTrain init slurm` 生成 `job.yaml`，其中包含工作流的所有可修改控制参数。  
Run `NepTrain init slurm` on the command line to generate `job.yaml`, which contains all modifiable control parameters of the workflow.  
会产生一个 `job.yaml` 文件，打开后可以看到默认的执行参数及其解释。首次运行时，可逐行设置参数；之后可复制该文件作为训练模板。  
The command produces a `job.yaml` file showing default execution parameters and their explanations. Set the parameters line by line on the first run, and later reuse the file as a template for future training.
:::{tip}
如果您复制了之前修改过的 `job.yaml`，在版本更新时可以将其复制到工作目录并运行 `NepTrain init slurm`，以同步新增参数。  
If you copy a previously modified `job.yaml`, you can place it in the working directory and run `NepTrain init slurm` again to synchronize any newly added parameters during version updates.
:::

### 开始训练 / Start Training
在登录节点的终端执行以下命令：  
Run the following command on the login node:
```shell
NepTrain train job.yaml
```
后台运行：  
To run in the background:
```shell
nohup NepTrain train job.yaml &
```
