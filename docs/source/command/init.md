# init
**Description:**  
Initializes file templates for NepTrain.
## Input Parameters

**Usage:**  
```bash
NepTrain init <type> [-f]
```

**Options:**
- `<type>`
  Job scheduler type. Choices: `bohrium`, `slurm`, `pbs`, `shell`. Default: `slurm`.
- `-f, --force`
  Force overwriting of generated templates. Default: `False`.

## Output
:::{tip}
These output files serve as inputs for the `train` command. Detailed modification instructions are provided in the [train](train.md) section.
::: 
- `job.yaml`  
  The configuration file for automatic training (`config_path`).  

- `run.in`  
  The template file for molecular dynamics (MD).  

- `structure/`
  This folder must contain the structures required for active learning. Multiple structures can be included, and the file format should be either `.xyz` or `.vasp`.
### 可选的模板文件
- INCAR
用户可以通过`INCAR`文件指定单点能的计算细节。如果没有指定则使用默认的INCAR。默认INCAR细节详见[INCAR](vasp.md)
- nep.in
如果没有指定该文件，会根据训练集自动判断元素种类，生成最简的nep.in。如果需要修改训练超参数，请自行创建该文件。
  最简的nep.in如下
    ```text
      generation     100000
      type     3 I Cs Pb
      ```
## 文件修改
生成的模板文件可以根据实际环境进行修改：

- 根据训练需求编辑 `job.yaml` 中的参数。
- 如需自定义训练超参数，可手动修改或创建 `nep.in`。

