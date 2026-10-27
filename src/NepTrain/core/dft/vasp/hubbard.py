"""Element-bound VASP Hubbard parameters and written-input identity audit."""
from collections.abc import Mapping
from pathlib import Path
import math

import numpy as np
from ase.data import atomic_numbers
from ase.io import read

from ...content_addressing import file_sha256


def validate_hubbard(mapping, symbols=None):
    if mapping is None:
        return None
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError('labeling.hubbard must be a nonempty element mapping')
    result = {}
    for element, values in mapping.items():
        if element not in atomic_numbers or element == 'X':
            raise ValueError(f'unknown Hubbard element: {element}')
        if not isinstance(values, Mapping) or set(values) != {'L', 'U', 'J'}:
            raise ValueError(f'hubbard.{element} requires exactly L, U, J')
        l = values['L']
        if isinstance(l, bool) or not isinstance(l, int) or l not in {-1, 0, 1, 2, 3}:
            raise ValueError(f'hubbard.{element}.L must be -1, 0, 1, 2 or 3')
        for key in ('U', 'J'):
            x = values[key]
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0:
                raise ValueError(f'hubbard.{element}.{key} must be finite and non-negative')
            if not math.isclose(x, round(x, 3), rel_tol=0, abs_tol=1e-12):
                raise ValueError(f'hubbard.{element}.{key}: ASE VASP writes three decimal places; refusing rounding')
        if l == -1 and (values['U'] != 0 or values['J'] != 0):
            raise ValueError(f'hubbard.{element}: disabled L=-1 requires U=J=0')
        result[element] = {'L': l, 'U': float(values['U']), 'J': float(values['J'])}
    if symbols is not None:
        missing = set(symbols) - set(result)
        if missing:
            raise ValueError('missing Hubbard elements: ' + ', '.join(sorted(missing)))
    return result


def configure_hubbard(calculator, mapping, symbols=None):
    normalized = validate_hubbard(mapping, symbols)
    if normalized is None:
        return None
    for key, params in [('ldaul', calculator.list_int_params),
                        ('ldauu', calculator.list_float_params), ('ldauj', calculator.list_float_params)]:
        if params.get(key) is not None:
            raise ValueError(f'labeling.hubbard conflicts with INCAR {key.upper()}; remove the positional list')
    if calculator.bool_params.get('ldau') is not True:
        raise ValueError('labeling.hubbard requires LDAU=.TRUE. in INCAR')
    if calculator.int_params.get('ldautype') not in (None, 2):
        raise ValueError('labeling.hubbard currently supports Dudarev LDAUTYPE=2 only')
    calculator.set(ldau_luj=normalized)
    return normalized


def audit_written_input(calculator, atoms):
    """Called after ASE writes inputs and before it launches any executable."""
    from ase.calculators.vasp import Vasp
    from .io import read_vasp_input
    from .deltaspin import numbers

    binding = calculator.input_binding
    root = Path(calculator.directory)
    order = np.asarray(binding['order'])[calculator.sort]
    expected = atoms[calculator.sort]
    written = read(root / 'POSCAR')
    if written.get_chemical_symbols() != expected.get_chemical_symbols():
        raise ValueError('written POSCAR atom order mismatch')
    if not np.allclose(written.positions, expected.positions, rtol=0, atol=1e-10):
        raise ValueError('written POSCAR positions mismatch')
    if not np.allclose(written.cell, expected.cell, rtol=0, atol=1e-10):
        raise ValueError('written POSCAR cell mismatch')
    if file_sha256(root / 'POTCAR') != binding['potcar_sha256']:
        raise ValueError('written POTCAR hash/order differs from pinned resources')
    parsed = Vasp()
    read_vasp_input(parsed, root / 'INCAR')
    mapping = binding['hubbard']
    species = [s for s, _ in calculator.symbol_count]
    actual = {}
    if mapping is not None:
        for key, field, params in [('LDAUL', 'L', parsed.list_int_params),
                                    ('LDAUU', 'U', parsed.list_float_params),
                                    ('LDAUJ', 'J', parsed.list_float_params)]:
            values = params.get(key.lower())
            if values is None or len(values) != len(species) or not np.allclose(values, [mapping[s][field] for s in species], rtol=0, atol=1e-12):
                raise ValueError(f'written {key} does not match element-bound Hubbard parameters')
            actual[key] = list(values)
    if binding['deltaspin']:
        if list(calculator.sort) != list(range(len(atoms))):
            raise ValueError('unexpected second ASE reordering of grouped DeltaSpin atoms')
        custom = parsed.input_params['custom']
        mask = numbers(custom['deltaspin_atoms'])
        if mask.shape != (len(atoms),) or not np.array_equal(mask, binding['active_mask']):
            raise ValueError('written DeltaSpin atom mask mismatch')
        for key, raw in [('MAGMOM', parsed.list_float_params['magmom']),
                         ('M_DELTASPIN', numbers(custom['m_deltaspin']))]:
            vectors = np.asarray(raw)
            if vectors.shape != (3 * len(atoms),) or not np.allclose(vectors.reshape(-1, 3), atoms.arrays['spin'], rtol=0, atol=1e-12):
                raise ValueError(f'written {key} atom/vector mismatch')
    return dict(hubbard_by_element=mapping, species=species,
                counts=[int(n) for _, n in calculator.symbol_count],
                expanded_hubbard=actual, vasp_to_original=order.tolist(),
                original_to_vasp=np.argsort(order).tolist(),
                input_sha256={name: file_sha256(root / name) for name in ['INCAR', 'POSCAR', 'POTCAR', 'KPOINTS'] if (root / name).is_file()})
