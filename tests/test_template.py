from NepTrain.core.template import get_job_config


def test_slurm_template_uses_controller_execution_targets_only():
    config = get_job_config("slurm")

    assert config["schema_version"] == 3
    assert set(config["execution"]["targets"]) == {"training", "cpu", "dft"}
    assert config["execution"]["routes"] == {
        "training": "training",
        "sampling": "cpu",
        "labeling": "dft",
        "analysis": "cpu",
    }
