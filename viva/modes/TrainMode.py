import argparse
import json
from pathlib import Path
from typing import Dict, List, Type

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import SimpleProfiler
from rich.console import Console
from torch.utils.data import DataLoader

from viva.modes.VivaBaseMode import VivaBaseMode
from viva.strategies.BaseTrainStrategy import BaseTrainStrategy, BaseTrainOptions
from viva.strategies.BlockStrategy import BlockStrategy

train_strategies: Dict[str, Type[BaseTrainStrategy]] = {
    "block": BlockStrategy
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
        options: BaseTrainOptions = strategy.options
        options.overwrite_options(training_overrides)

        self.console.print(options)

        # Load datasets
        data = json.loads(dataset_path.read_text(encoding="utf-8"))

        train_dataset = strategy.train_dataset_type(metadata_paths=data["train"], block_length=options.block_size)
        val_dataset = strategy.test_dataset_type(metadata_paths=data["val"], block_length=options.block_size)

        x, y = train_dataset[0]
        self.console.print(f"Data X Shape: {x.shape}")
        self.console.print(f"Data Y Shape: {y.shape}")

        # Prepare output paths
        log_dir = Path(options.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=options.batch_size, shuffle=True,
                                  num_workers=options.num_workers, persistent_workers=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=options.batch_size, shuffle=False,
                                num_workers=options.num_workers, persistent_workers=True, pin_memory=True)

        # Create Model
        model = strategy.create_lighting_module()

        # TensorBoard Logger
        logger = TensorBoardLogger(save_dir=log_dir, name=model.__class__.__name__)

        if options.early_stopping:
            early_stopping = EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=options.early_stopping_patience,
                verbose=True
            )
        else:
            early_stopping = None

        # Define Model Checkpoints
        checkpoint_callback = ModelCheckpoint(
            filename="best",
            monitor="val_loss",  # Metric to monitor
            mode="min",  # Mode can be 'min' for loss or 'max' for accuracy/metrics
            save_last=True,  # Save the last epoch's model
            save_top_k=1,  # Save the best model
            verbose=True
        )

        # create profiler if requested
        profiler = SimpleProfiler() if options.profile else None
        precision = 16 if options.mixed else None

        # Trainer
        trainer = Trainer(
            max_epochs=options.max_epochs,
            logger=logger,
            log_every_n_steps=50,
            profiler=profiler,
            callbacks=[checkpoint_callback, early_stopping],
            precision=precision
        )

        # Train the model
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Save additional information about the best model
        best_model_path = checkpoint_callback.best_model_path
        print(f"Best model saved at {best_model_path}")

        # Test the model (if needed)
        # test_loader = DataLoader(test_dataset, batch_size=options.batch_size, shuffle=False,
        #                          num_workers=options.num_workers, persistent_workers=True)
        # trainer.test(model, test_loader)

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
