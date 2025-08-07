# gpumd

**Description:**  
Runs molecular dynamics simulations using GPUMD.
## Input Parameters

**Usage:**  
```bash
NepTrain gpumd <model_path> [options]
```
:::{tip}
For simplicity in execution and to reduce input parameters, `./run.in` and `./nep.txt` are used as the default input files. However, these file paths must be valid.  
If the filenames are incorrect, please specify them using the appropriate parameters.
::: 
**Options:**  
- `<model_path>`  
  Structure path or file (supports `xyz` and `vasp` formats).
- `-dir, --directory`  
  Path for GPUMD calculations. Default: `./cache/gpumd`.
- `-in, --in`  
  Path to `run.in` file. Default: `./run.in`.
- `-nep, --nep`  
  Path to potential function file. Default: `./nep.txt`.
- `-t, --time`  
  Molecular dynamics simulation time (ps). Default: `10`.
- `-T, --temperature`
  Simulation temperature(s) (K). Default: `300`.
- `-o, --out`
  Output file for trajectory. Default: `./trajectory.xyz`.

## Output
GPUMD will output files in the `directory` (default is `./cache/gpumd`) along with GPUMD's standard output files. The resulting trajectories are saved to the path specified by `--out`.



## Example

:::{tip}
Direct execution will run on the local machine. If you need to submit the job to a queueing system, place this command at the end of the submission script.
:::

### Minimal Execution Command
If `./run.in` and `./nep.txt` already exist in the directory, and your structure files are stored in the `structure` folder, you can run the following command directly from the directory. This will perform a 10 ps molecular dynamics (MD) simulation at 300 K for all structure files in the `structure` folder:

```shell
NepTrain gpumd structure
```

### Modifying Parameters
To run MD simulations for all structures at temperatures ranging from 100 K to 600 K with a step of 100 K, you can use either of the following commands:

```shell
NepTrain gpumd structure -T 100 200 300 400 500 600 
```

Or use a shell shortcut to create an array:

```shell
NepTrain gpumd structure -T {100...600...100}
```
 
