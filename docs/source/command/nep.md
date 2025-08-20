# nep
**Description:**  
Trains potential functions using NEP.
## Input Parameters

**Usage:**  
```bash
NepTrain nep [options]
```

**Options:**  
- `-dir, --directory`  
  Path for NEP calculations. Default: `./cache/nep`.
- `-in, --in`  
  Path to `nep.in` file. Default: `./nep.in`.
- `-train, --train`  
  Path to `train.xyz` file. Default: `./train.xyz`.
- `-test, --test`  
  Path to `test.xyz` file. Default: `./test.xyz`.
- `-nep, --nep`  
  Path to potential function file. Default: `./nep.txt`.
- `-pred, --prediction, --pred`
  Enable prediction mode. Default: `False`.
- `-restart, --restart_file, --restart`
  Path to restart file. Default: `None`.
- `-cs, --continue_step`
  Steps to continue from restart. Default: `10000`.
## Output
Training results are written to the directory specified by `--directory`
(default `./cache/nep`). Important files include:

- `nep.txt` – trained potential function.
- `loss.out` – training loss for each step.
- `nep.out` / `nep.err` – standard output and error logs.
- `nep_result.png` – plot of the training loss.

## Example
Run NEP training with explicit training and testing datasets:

```bash
NepTrain nep --train train.xyz --test test.xyz
```

Resume training from a restart file for an additional 50,000 steps:

```bash
NepTrain nep --restart_file ./cache/nep/nep.restart --continue_step 50000
```
