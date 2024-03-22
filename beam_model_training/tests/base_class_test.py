from pathlib import PosixPath
import pytest
import yaml

from utils.base_class import BaseClass

base_config = {"seed": 2022, "train": {"finetune": False}}

test_cases = {
    "single_model": {
        "models": 1,
        "config": base_config,
        "expected": {"success": True, "parent": "models"},
    },
    "double_model": {
        "models": 2,
        "config": base_config,
        "expected": {"success": False},
    },
    "double_model_with_model_name": {
        "models": 2,
        "config": {**base_config, "model_version": "test_run_1"},
        "expected": {"success": True, "parent": "models"},
    },
    "model_and_base_model_finetune": {
        "models": 1,
        "base_model": 1,
        "config": {**base_config, "train": {"finetune": True}},
        "expected": {"success": True, "parent": "base_model"},
    },
    "model_and_base_model_no_finetune": {
        "models": 1,
        "base_model": 1,
        "config": base_config,
        "expected": {"success": True, "parent": "models"},
    },
    "model_and_double_base_model_finetune": {
        "models": 1,
        "base_model": 2,
        "config": {**base_config, "train": {"finetune": True}},
        "expected": {"success": False},
    },
    "model_and_double_base_model_finetune_with_model_version": {
        "models": 1,
        "base_model": 2,
        "config": {
            **base_config,
            "train": {"finetune": True},
            "model_version": "test_run_1",
        },
        "expected": {"success": True, "parent": "base_model"},
    },
}


class Test_BaseClass:

    @pytest.fixture(scope="class", params=test_cases.values(), ids=test_cases.keys())
    def base_class_and_expected(self, request, tmp_path_factory):
        params = request.param
        test_dir = tmp_path_factory.mktemp("base_class")

        test_dir.mkdir(parents=True, exist_ok=True)
        with open(test_dir / "test_config.yaml", "w") as file:
            yaml.dump(params["config"], file)
        for file in test_dir.iterdir():
            print(file)
        print(params["config"])

        # Create models directory
        models_dir = test_dir / BaseClass.DIR_STRUCTURE["models"]
        for i in range(params.get("models", 0)):

            test_run_dir = models_dir / f"test_run_{i}"
            test_run_dir.mkdir(parents=True, exist_ok=True)
            mock_pkl = test_run_dir / "fake_model.pkl"
            mock_pkl.touch()

        # Create base model directory
        base_model_dir = test_dir / BaseClass.DIR_STRUCTURE["base_model"]
        for i in range(params.get("base_model", 0)):

            test_run_dir = base_model_dir / f"test_run_{i}"
            test_run_dir.mkdir(parents=True, exist_ok=True)
            mock_pkl = test_run_dir / "fake_model.pkl"
            mock_pkl.touch()

        base_class = BaseClass(test_dir)
        base_class.load_dir_structure(read_dirs=["models", "base_model"])
        yield base_class, params["expected"]

    def test_load_model(self, base_class_and_expected):
        base_class, expected = base_class_and_expected
        if expected["success"]:
            model_path = base_class.load_model_path(
                base_class.config.get("model_version"),
                base_class.config["train"]["finetune"],
            )
            assert isinstance(
                model_path,
                PosixPath,
            ), "The load_model_path function should return a PosixPath"
            assert (
                model_path.parent.parent.name == expected["parent"]
            ), f"The run directory is expected to be found in `{expected['parent']}`."
        else:
            with pytest.raises(FileNotFoundError):
                base_class.load_model_path(
                    base_class.config.get("model_version"),
                    base_class.config["train"]["finetune"],
                )
