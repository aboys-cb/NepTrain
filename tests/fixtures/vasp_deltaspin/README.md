# VASP 6 DeltaSpin parser fixture

Extracted from `sai:/org/caep-xuben/share/examples/vasp-deltaspin/OUTCAR`
on 2026-09-29. Only the final status and magnetic tables are retained.
The accompanying INCAR and POSCAR describe that run (Fe2, SOC, moment definition 0).
No POTCAR or VASP source is included.

Full source OUTCAR SHA256: `ffcd9b83249c9afd6fe0d60649980d6a0959edcd24fada60ad5f1df27232dee8`.

Both convergence flags are reached; maximum moment error is 8.0856e-8 uB
against a tolerance of 1e-6 uB. Tests compare the printed values without
changing the magnetic-force sign. Synthetic cases derived from this format
test malformed/partial output and multi-element ordering; they do not validate
VASP dynamics or finite-difference derivatives.
