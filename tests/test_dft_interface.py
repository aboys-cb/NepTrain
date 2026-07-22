from pathlib import Path
from types import SimpleNamespace

from NepTrain.core import dft as dft_module


def test_direct_dft_command_forwards_resource_directory(tmp_path, monkeypatch):
    resources = tmp_path / "resources"
    resources.mkdir()
    captured = {}

    def fake_label(request, backend):
        captured["request"] = request
        captured["backend"] = backend
        return "labeled"

    monkeypatch.setattr(dft_module, "label", fake_label)
    result = dft_module.run_dft(
        SimpleNamespace(
            software="vasp",
            model_path="selected.xyz",
            out_file_path="labeled.xyz",
            directory="teacher",
            append=False,
            incar="INCAR",
            resource_dir=str(resources),
            n_cpu=1,
            use_gamma=True,
            kspacing=0.2,
            ka=[1, 1, 1],
            teacher_profile="ordinary",
        )
    )

    assert result == "labeled"
    assert captured["backend"] == "vasp"
    assert captured["request"].resource_dir == resources
    assert captured["request"].input_file == Path("INCAR")
    assert captured["request"].kpoint_mode == "kspacing"
