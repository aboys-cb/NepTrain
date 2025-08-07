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
### 可选的模板文件 / Optional Template Files
- INCAR or INPUT  
  用户可以通过 `INCAR` 或 `INPUT` 文件指定单点能计算的细节。如果没有提供，将使用默认设置。默认的 INCAR 参数详见 [INCAR](vasp.md)。  
  Users can specify details of single-point energy calculations through the `INCAR` or `INPUT` file. If not provided, the defaults are used. For the default INCAR settings, see [INCAR](vasp.md).
- nep.in  
  如果没有提供该文件，程序会根据训练集自动识别元素种类，并生成最简的 `nep.in`。若需修改训练超参数，请自行创建此文件。  
  If this file is absent, the program automatically detects element types from the training set and generates a minimal `nep.in`. Create this file manually to modify training hyperparameters.  
  最简的 `nep.in` 如下  
  The minimal `nep.in` is:
    ```text
      generation     100000
      type     3 I Cs Pb
    ```
## 文件修改 / File Modification
生成的模板文件可以根据实际环境进行调整：  
The generated template files can be adapted to your environment:

- 根据训练需求编辑 `job.yaml` 中的参数。  
  Edit the parameters in `job.yaml` according to your training requirements.
- 如需自定义训练超参数，可手动修改或创建 `nep.in`。  
  To customize training hyperparameters, manually modify or create `nep.in`.

