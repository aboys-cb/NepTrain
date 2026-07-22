"""NEPAdapters-backed calculation and descriptor compatibility interface."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from ase import Atoms


def _nep_adapters():
    try:
        import nep_adapters
    except ImportError as error:  # pragma: no cover - installation failure path
        raise RuntimeError(
            "NEPAdapters is required. Install a matching nep-adapters wheel."
        ) from error
    return nep_adapters


def resolve_backend(model_file: str | Path, requested: str = "auto") -> str:
    """Resolve auto/cpu/cuda without silently changing explicit choices."""

    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("backend must be auto, cpu, or cuda")
    nep_adapters = _nep_adapters()
    if requested != "auto":
        status = nep_adapters.backend_status(requested)
        if not status.available:
            raise RuntimeError(f"NEPAdapters {requested} backend unavailable: {status.detail}")
        # Loading is the model-capability gate. Explicit choices fail closed.
        with nep_adapters.NEPCalculator(model_file, backend=requested):
            pass
        return requested

    status = nep_adapters.backend_status("cuda")
    if status.available:
        try:
            with nep_adapters.NEPCalculator(model_file, backend="cuda"):
                pass
            return "cuda"
        except nep_adapters.NepAdaptersError:
            pass
    return "cpu"


class Nep3Calculator:
    """Compatibility facade whose implementation is entirely NEPAdapters."""

    def __init__(self, model_file: str | Path = "nep.txt", backend: str = "auto"):
        nep_adapters = _nep_adapters()
        self.backend = resolve_backend(model_file, backend)
        self._calculator = nep_adapters.NEPCalculator(model_file, backend=self.backend)
        self.model_info = self._calculator.model_info
        self.element_list = list(self.model_info.elements)
        self.type_dict = dict(self._calculator.type_dict)

    def close(self) -> None:
        self._calculator.close()

    def __enter__(self) -> "Nep3Calculator":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    @staticmethod
    def _structures(structures: Iterable[Atoms] | Atoms) -> list[Atoms]:
        if isinstance(structures, Atoms):
            return [structures]
        return list(structures)

    def get_descriptors(self, structure: Atoms) -> np.ndarray:
        if self.model_info.supports("spin"):
            return self._calculator.get_spin_descriptor(structure)
        return self._calculator.get_descriptor(structure)

    def get_structure_descriptors(self, structure: Atoms) -> np.ndarray:
        return self.get_descriptors(structure).mean(axis=0)

    def get_structures_descriptors(self, structures: Iterable[Atoms]) -> np.ndarray:
        frames = self._structures(structures)
        if self.model_info.supports("spin"):
            return self._calculator.get_spin_structures_descriptor(frames)
        return self._calculator.get_structures_descriptor(frames)

    def calculate(self, structures: Iterable[Atoms] | Atoms, mean_virial: bool = True):
        frames = self._structures(structures)
        if self.model_info.supports("spin"):
            prediction = self._calculator.predict_spin_structures(frames)
        else:
            prediction = self._calculator.predict_structures(frames)
        return (
            prediction.energy,
            prediction.force_blocks(),
            prediction.virial_blocks(mean=mean_virial),
        )

    def calculate_spin(
        self, structures: Iterable[Atoms] | Atoms, mean_virial: bool = True
    ):
        prediction = self._calculator.predict_spin_structures(self._structures(structures))
        return (
            prediction.energy,
            prediction.force_blocks(),
            prediction.virial_blocks(mean=mean_virial),
            prediction.mforce_blocks(),
        )


class DescriptorCalculator:
    def __init__(self, calculator_type: str = "nep", **calculator_kwargs):
        self.calculator_type = calculator_type
        if calculator_type == "nep":
            self.calculator = Nep3Calculator(**calculator_kwargs)
        elif calculator_type == "soap":
            from dscribe.descriptors import SOAP

            self.calculator = SOAP(**calculator_kwargs, dtype="float32")
        else:
            raise ValueError("calculator_type must be nep or soap")

    def get_structures_descriptors(self, structures: Iterable[Atoms]) -> np.ndarray:
        frames = list(structures)
        if not frames:
            return np.array([])
        if self.calculator_type == "nep":
            return self.calculator.get_structures_descriptors(frames)
        return np.asarray([self.calculator.create_single(frame).mean(0) for frame in frames])
