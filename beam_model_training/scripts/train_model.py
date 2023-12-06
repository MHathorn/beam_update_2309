
import os

from segmentation.train import Trainer
from utils.helpers import load_config


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"

    config = load_config("base_config.yaml")
    trainer = Trainer(config)
    trainer.run()

    


if __name__ == '__main__':
    main()