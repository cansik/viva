import argparse
import json
from pathlib import Path
from typing import Dict, List, Type

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from rich.console import Console
from torch.utils.data import DataLoader

from viva.modes.VivaBaseMode import VivaBaseMode
from viva.strategies.BaseTrainStrategy import BaseTrainStrategy
from viva.strategies.BlockStrategy import BlockStrategy

train_strategies: Dict[str, Type[BaseTrainStrategy]] = {
    "block-tcn": BlockStrategy
}


class TrainMode(VivaBaseMode):
    def __init__(self, console: Console):
        super().__init__(console)

    def run(self):
        args = self._parse_args()

        strategy = train_strategies[args.strategy]()
        dataset_path = Path(args.dataset)
        training_overrides: List[str] = args.training_overrides

        # Create options and override them
        options = strategy.options
        options.overwrite_options(training_overrides)

        # Load datasets
        data = json.loads(dataset_path.read_text(encoding="utf-8"))

        train_dataset = strategy.dataset_type(metadata_paths=data["train"], block_length=options.block_size)
        test_dataset = strategy.dataset_type(metadata_paths=data["test"], block_length=options.block_size)
        val_dataset = strategy.dataset_type(metadata_paths=data["val"], block_length=options.block_size)

        x, y = train_dataset[0]
        self.console.print(f"Data X Shape: {x.shape}")
        self.console.print(f"Data Y Shape: {y.shape}")

        # Prepare output paths
        log_dir = Path(options.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=options.batch_size, shuffle=True,
                                  num_workers=options.num_workers, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=options.batch_size, shuffle=False,
                                num_workers=options.num_workers, persistent_workers=True)

        # Create Model
        model = strategy.create_lighting_module()

        # TensorBoard Logger
        logger = TensorBoardLogger(save_dir=log_dir, name=model.__class__.__name__)

        # Trainer
        trainer = Trainer(max_epochs=options.max_epochs, logger=logger, log_every_n_steps=50)

        # Train the model
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Test the model
        # test_loader = DataLoader(test_dataset, batch_size=options.batch_size, shuffle=False,
        #                          num_workers=options.num_workers, persistent_workers=True)
        # trainer.test(model, test_loader)

        # Save the trained model
        model_path = Path(options.log_dir) / f"{model.__class__.__name__}.ckpt"
        trainer.save_checkpoint(model_path)

        print(f"Model {model.__class__.__name__} saved at {model_path}")
        print(f"Logs are available at {logger.log_dir}")

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(prog="viva train")
        parser.add_argument("dataset", type=str, help="Path to the dataset file.")
        parser.add_argument("strategy", choices=train_strategies.keys(),
                            help="Which training strategy to use for training.")
        parser.add_argument("training_overrides", type=str, nargs="*",
                            help="Arguments to overwrite training options (name=value).")
        return parser.parse_args()
