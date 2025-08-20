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
### Optional Template Files
- INCAR or INPUT  
  Users can specify details of single-point energy calculations through the `INCAR` or `INPUT` file. If not provided, the defaults are used. For the default INCAR settings, see [INCAR](vasp.md).
- nep.in  
  If this file is absent, the program automatically detects element types from the training set and generates a minimal `nep.in`. Create this file manually to modify training hyperparameters.  
  The minimal `nep.in` is:
    ```text
      generation     100000
      type     3 I Cs Pb
    ```
## File Modification
The generated template files can be adapted to your environment:

  - Edit the parameters in `job.yaml` according to your training requirements.
  - To customize training hyperparameters, manually modify or create `nep.in`.

