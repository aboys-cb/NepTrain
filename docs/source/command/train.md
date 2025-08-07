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
 
### 初始化操作
在命令行输入`NepTrain init slurm`产生`job.yaml`，包括工作流中的所有控制参数，可修改。
将会产生一个job.yaml文件，打开这个文件，里边将会显示默认的执行参数以及对每个参数的解释，
在首次运行时，你可能需要逐行对参数进行设置，在以后运行时，你可以复制这个job.yaml文件到以后运行的工作目录作为训练的模板文件。
:::{tip}
如果您拷贝之前修改后的job.yaml。如果涉及版本变化，可以拷贝到工作路径，后执行`NepTrain init slurm`。会将新加入的参数同步过来！
:::
 
### 开始训练
在登陆节点的终端执行一下命令
```shell
NepTrain train job.yaml
```
后台运行
```shell
nohup NepTrain train job.yaml &
```
