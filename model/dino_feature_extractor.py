import enum
import torch
import torch.nn as nn

from model.label_generators.modules import DINO


class DINOFeatureExtractor(enum.Enum):
    DINO = enum.auto()

    def get_model(self, pretrained_DINO_path):
        return self._get_config(self)(pretrained_DINO_path)

    def _get_config(self, value):
        return {
            self.DINO: self._get_dino,
        }[value]
    
    def _get_dino(self, pretrained_DINO_path):
        return DINO(pretrained_DINO_path)