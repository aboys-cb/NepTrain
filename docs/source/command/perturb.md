# perturb
**Description:**  
Generates perturbed structures.
## Input Parameters

**Usage:**  
```bash
NepTrain perturb <model_path> [options]
```

**Options:**  
- `<model_path>`  
  The structure path or file for calculation (supports `xyz` and `vasp` formats).
- `-n, --num`  
  Number of perturbations for each structure. Default: `20`.
- `-c, --cell`  
  Deformation ratio. Default: `0.03`.
- `-d, --distance`
  Minimum atom distance (Å). Default: `0.1`.
- `-o, --out`
  Output file path for perturbed structures. Default: `./perturb.xyz`.
- `-a, --append`
  Append to output file instead of overwriting. Default: `False`.
## Output
All generated structures are written to the file specified by `--out`
(default `./perturb.xyz`). When `--append` is used, new structures are
appended to the existing file instead of overwriting it.

## Example
Generate 2000 perturbed configurations of a VASP structure file with a
cell distortion of `0.03` and a minimum atomic distance of `0.1 Å`:

```bash
NepTrain perturb ./structure/Cs16Ag8Bi8I48.vasp --num 2000 --cell 0.03 -d 0.1
```
