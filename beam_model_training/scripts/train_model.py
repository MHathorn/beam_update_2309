from pathlib import Path

from training.training_functions import check_dataset_balance
from training.train import train
import os
import yaml

from utils.helpers import create_if_not_exists, seed, timestamp, load_config



def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"

    config = load_config("base_config.yaml")
    seed(config["seed"])

    path = Path(config["root_dir"])

    model_dir = create_if_not_exists(path / config["dirs"]["models"])
    train_dir = path / config["dirs"]["train"]

    # percentages = check_dataset_balance()


    learner = train(train_dir, model_dir, tile_size=config["tile_size"], split=config["test_size"], **config["learn"])
    ts = timestamp()
    model_path = model_dir / f"model_{ts}.pkl"
    learner.export(model_path)

    with open(model_dir / f'config_{ts}.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


if __name__ == '__main__':
    main()