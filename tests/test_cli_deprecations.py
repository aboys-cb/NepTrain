from types import SimpleNamespace

from NepTrain.cli import cli


def test_vasp_compatibility_command_recommends_dft(monkeypatch, capsys):
    called = []
    monkeypatch.setattr("NepTrain.core.dft.vasp.run_vasp", called.append)
    args = SimpleNamespace()

    cli.run_vasp(args)

    assert called == [args]
    assert capsys.readouterr().err == (
        "NepTrain: warning: 'vasp' is a compatibility command and will be "
        "removed in the next release; use 'dft --vasp' instead.\n"
    )


def test_gpumd_compatibility_command_recommends_md(monkeypatch, capsys):
    called = []
    monkeypatch.setattr("NepTrain.core.gpumd.run_gpumd", called.append)
    args = SimpleNamespace()

    cli.run_gpumd(args)

    assert called == [args]
    assert capsys.readouterr().err == (
        "NepTrain: warning: 'gpumd' is a compatibility command and will be "
        "removed in the next release; use 'md --backend gpumd' instead.\n"
    )
