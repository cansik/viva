from dataclasses import dataclass
from functools import partial
from typing import Callable, Type, Union, Dict

import pytorch_lightning as pl
from visiongraph import vg

from viva.data.FaceLandmarkDataset import FaceLandmarkDataset
from viva.data.augmentations.CollapseLabels import CollapseLabels
from viva.data.augmentations.FilterLandmarkIndices import FilterLandmarkIndices
from viva.data.augmentations.FlattenLandmarks import FlattenLandmarks
from viva.data.augmentations.NormalizeLandmarks import NormalizeLandmarks
from viva.data.augmentations.OneHotEncodeLabels import OneHotEncodeLabels
from viva.data.augmentations.RandomNoiseAugmentation import RandomNoiseAugmentation
from viva.models.ImprovedTCNLandmarkClassifier import ImprovedTCNLandmarkClassifier
from viva.models.LSTMLandmarkClassifier import LSTMLandmarkClassifier
from viva.models.SimpleMLPClassifier import SimpleMLPClassifier
from viva.models.TCNLandmarkClassifier import TCNLandmarkClassifier
from viva.models.TransformerLandmarkClassifier import TransformerLandmarkClassifier
from viva.strategies.BaseTrainStrategy import BaseTrainStrategy, T, BaseTrainOptions

INPUT_SIZE_FEATURES_148 = len(vg.BlazeFaceMesh.FEATURES_148) * 3


@dataclass
class NetworkConfig:
    network_factory: Callable[["BlockStrategyOptions"], pl.LightningModule]
    flatten_full: bool = False


BLOCK_STRATEGY_NETWORKS: Dict[str, NetworkConfig] = {
    "tcn": NetworkConfig(
        lambda _: TCNLandmarkClassifier(input_size=INPUT_SIZE_FEATURES_148)
    ),
    "tcn-i": NetworkConfig(
        lambda _: ImprovedTCNLandmarkClassifier(INPUT_SIZE_FEATURES_148)
    ),
    "mlp": NetworkConfig(
        lambda x: SimpleMLPClassifier(input_size=x.block_length * INPUT_SIZE_FEATURES_148),
        flatten_full=True
    ),
    "transformer": NetworkConfig(
        lambda x: TransformerLandmarkClassifier(input_size=INPUT_SIZE_FEATURES_148)
    ),
    "lstm": NetworkConfig(
        lambda x: LSTMLandmarkClassifier(input_size=INPUT_SIZE_FEATURES_148)
    )
}


@dataclass
class BlockStrategyOptions(BaseTrainOptions):
    network: str = "tcn"
    noise_augmentation: bool = False


class BlockStrategy(BaseTrainStrategy[BlockStrategyOptions]):
    def __init__(self):
        self._options = BlockStrategyOptions()

    @property
    def options(self) -> T:
        return self._options

    @property
    def network_config(self) -> NetworkConfig:
        return BLOCK_STRATEGY_NETWORKS[self._options.network]

    def create_lighting_module(self) -> pl.LightningModule:
        return self.network_config.network_factory(self._options)

    @property
    def train_dataset_type(self) -> Union[Type[FaceLandmarkDataset], Callable[..., FaceLandmarkDataset]]:
        transforms = [
            NormalizeLandmarks(),
            FilterLandmarkIndices(vg.BlazeFaceMesh.FEATURES_148)
        ]

        augmentations = [
            FlattenLandmarks(full=self.network_config.flatten_full),
            CollapseLabels(),
            OneHotEncodeLabels()
        ]

        if self._options.noise_augmentation:
            augmentations.insert(0, RandomNoiseAugmentation(0.1, clip=True))

        return partial(FaceLandmarkDataset, transforms=transforms, augmentations=augmentations)

    @property
    def test_dataset_type(self) -> Union[Type[FaceLandmarkDataset], Callable[..., FaceLandmarkDataset]]:
        transforms = [
            NormalizeLandmarks(),
            FilterLandmarkIndices(vg.BlazeFaceMesh.FEATURES_148)
        ]

        augmentations = [
            FlattenLandmarks(full=self.network_config.flatten_full),
            CollapseLabels(),
            OneHotEncodeLabels()
        ]

        return partial(FaceLandmarkDataset, transforms=transforms, augmentations=augmentations)
