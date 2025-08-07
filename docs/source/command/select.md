# select
**Description:**  
Selects samples from trajectory files.
## Input Parameters

**Usage:**
```bash
NepTrain select <trajectory_paths> [options]
```

**Options:**
- `<trajectory_paths>`
  One or more trajectory files (xyz format).
- `-base, --base`
  Path to `base.xyz` for sampling. Default: `train.xyz`.
- `-nep, --nep`  
  Path to `nep.txt` file for descriptor extraction. Default: `./nep.txt`.
- `-max, --max_selected`
  Maximum number of structures to select. Default: `20`.
- `-d, --min_distance`
  Minimum bond length for farthest-point sampling. Default: `0.01`.
- `-f, --filter [COEF]`
  Filter structures based on covalent radius. Filtering is disabled by default.
  When enabled, structures with bonds shorter than `COEF ×` the covalent radius are
  written to `remove_by_bond_structures.xyz`. If no coefficient is provided,
  it defaults to `0.6`.
- `--pca, -pca`
  Use PCA for decomposition.
- `--umap, -umap`
  Use UMAP for decomposition.
- `-o, --out`  
  Output file for selected structures. Default: `./selected.xyz`.
- **SOAP Parameters:**
  - `-r, --r_cut`
    Cutoff for local region (Å). Default: `6`.
  - `-n, --n_max`
    Number of radial basis functions. Default: `8`.
  - `-l, --l_max`
    Maximum degree of spherical harmonics. Default: `6`.
## Output
Selected structures are saved to the file specified by `--out`
(`./selected.xyz` by default). If `--filter` is provided, structures that
violate the bond-length criterion are additionally written to
`remove_by_bond_structures.xyz`.

## Example
Select up to 100 structures from a trajectory while filtering short
bonds and save the results to `selected.xyz`:

```bash
NepTrain select trajectory.xyz -max 100 -f -o selected.xyz
```
