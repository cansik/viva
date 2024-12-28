from dataclasses import dataclass
from functools import partial
from typing import Callable, Type, Union

import pytorch_lightning as pl
from visiongraph import vg

from viva.data.FaceLandmarkDataset import FaceLandmarkDataset
from viva.data.augmentations.CollapseLabels import CollapseLabels
from viva.data.augmentations.FilterLandmarkIndices import FilterLandmarkIndices
from viva.data.augmentations.FlattenLandmarks import FlattenLandmarks
from viva.data.augmentations.NormalizeLandmarks import NormalizeLandmarks
from viva.data.augmentations.OneHotEncodeLabels import OneHotEncodeLabels
from viva.models.TCNLandmarkClassifier import TCNLandmarkClassifier
from viva.strategies.BaseTrainStrategy import BaseTrainStrategy, T, BaseTrainOptions


@dataclass
class BlockStrategyOptions(BaseTrainOptions):
    pass


class BlockStrategy(BaseTrainStrategy[BlockStrategyOptions]):
    def __init__(self):
        self._options = BlockStrategyOptions()

    @property
    def options(self) -> T:
        return self._options

    def create_lighting_module(self) -> pl.LightningModule:
        # return SimpleMLPClassifier(input_size=15 * len(vg.BlazeFaceMesh.FEATURES_148) * 3)
        return TCNLandmarkClassifier(input_size=len(vg.BlazeFaceMesh.FEATURES_148) * 3)

    @property
    def dataset_type(self) -> Union[Type[FaceLandmarkDataset], Callable[..., FaceLandmarkDataset]]:
        transforms = [
            NormalizeLandmarks(),
            FilterLandmarkIndices(vg.BlazeFaceMesh.FEATURES_148)
        ]

        augmentations = [
            FlattenLandmarks(full=False),
            CollapseLabels(),
            OneHotEncodeLabels()
        ]

        return partial(FaceLandmarkDataset, transforms=transforms, augmentations=augmentations)
