<div align="center"><strong>English</strong> | <a href="README.md">简体中文</a></div>

# Fe DeltaSpin, LAMMPS NPT, and multiple routes

This example covers constrained moments, spin NPT, and independent route coverage using
16-atom bcc Fe. Collinear and canted starts are tutorial inputs, not an accepted Fe model.

You need a DeltaSpin-enabled `vasp_ncl`, TorchNEP supporting `spin_mode 1`, matching
NEPAdapters, and LAMMPS with USER-DYNSPIN and the NEP plugin. The local process target uses
your current compute allocation; do not launch it on a cluster login node. For Slurm targets,
see the [ordinary VASP tutorial](../workflow-vasp-slurm/README.en.md).

```bash
cd examples/workflow-vasp-deltaspin
python -m pip install -e '../..[torchnep]'
python make_candidates.py
cp vasp-resources.example.json vasp-resources.json
```

Fill in the POTCAR root in `project.yaml` and the real Fe/POTCAR SHA256, TITEL, and release
in the manifest. Configure the LAMMPS plugin and, if needed, `NEPTRAIN_VASP_COMMAND`.
The generator writes 12 unlabeled structures and `structures/fm.xyz` / `structures/canted.xyz`.
It supplies per-atom `spin:R:3` in mu_B, never fabricated energy, forces, or magnetic forces.
Review functional, SOC, k points, and moment definition in INCAR before using them consistently.
NepTrain fills `MAGMOM` and `M_DELTASPIN` from each frame's spin targets.

```bash
neptrain label structures/canted.xyz --backend vasp --project project.yaml --wait --output check.xyz
python - <<'PY'
from ase.io import read
from NepTrain.core.spin import validate_spin_dataset
frames = read('check.xyz', index=':')
print(validate_spin_dataset(frames, require_mforce=True))
print(frames[0].arrays['spin'])
print(frames[0].arrays['mforce'])
PY
neptrain label seed-train.xyz --backend vasp --project project.yaml --wait --output train.xyz
neptrain doctor --project project.yaml
neptrain workflow run project.yaml --prepare-only
neptrain workflow run fe-deltaspin-workflow
neptrain workflow status fe-deltaspin-workflow --jobs
```

No test dataset is required. `nep.in` explicitly selects Spin NEP Lite (`spin_mode 1`);
training and inference must support the same format. The `dynspin/glsd/npt` template uses
LAMMPS metal units: K, bar, and ps. GPUMD pressure values in GPa are not automatically converted.

Inspect `generations/0001/label/` for spin/mforce labels and provenance, `evaluate/` for
pre-training E/F/M errors, and `update/` for the enlarged dataset. Status explains missing
accuracy, passing streaks, or production coverage independently for both routes.

One generation, 10–80 MD steps, and illustrative thresholds only test the interface.
`budget_exhausted` is expected and does not mean DeltaSpin failed or the model is accepted.
Production requires broader structural and spin coverage, derivative/sign checks, consistent
moment definitions, and NPT stability validation. Automated tests cover configuration,
generation, and rendering; VASP, LAMMPS, and spin training were not run in this change.
