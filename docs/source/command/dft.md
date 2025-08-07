# dft
**Description:**
Calculates single-point energies using external DFT programs such as VASP or ABACUS.

## Input Parameters
**Usage:**
```bash
NepTrain dft <model_path> [options]
```
**Options:**
- `<model_path>`
  Structure path or file (supports `xyz` and `vasp` formats).
- `-dir, --directory`
  Directory for DFT calculations. Default: `./cache/<software>`.
- `-o, --out`
  Output file path for calculated structure. Default: `./<software>_scf.xyz`.
- `-a, --append`
  Append to output file. Default: `False`.
- `-g, --gamma`
  Use Gamma-centered k-point scheme. Default: `False`.
- `-n, -np`
  Number of CPU cores. Default: `1`.
- `--in`
  Path to INCAR/INPUT file. Default: `./INCAR` for VASP, `./INPUT` for ABACUS.
- `-kspacing, --kspacing`
  Set k-spacing value.
- `-ka`
  Set k-points as 1 or 3 numbers (comma-separated). Default: `[1, 1, 1]`.
- `--vasp`
  Use VASP for the calculation (default).
- `--abacus`
  Use ABACUS for the calculation.

## Output
The selected DFT program is invoked to perform single-point energy calculations. Results and standard output are stored in `--directory` (default `./cache/<software>`). Processed structures are written to the file specified by `--out`.

## Example
To calculate single-point energies for all structures in the `structure` folder using the default VASP backend:
```shell
NepTrain dft structure -g
```
To run the same task with ABACUS and a custom input template:
```shell
NepTrain dft structure --abacus --in ./INPUT
```
