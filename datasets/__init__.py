import enum

from .fivr import FIVR
from .evve import EVVE
from .svd import SVD
from .generators import *


class EvaluationDataset(enum.Enum):
    FIVR_5K = enum.auto()
    FIVR_200K = enum.auto()
    EVVE = enum.auto()
    SVD = enum.auto()

    def get_dataset(self):
        return self._get_config(self)

    def get_eval_fn(self, input_type):
        return self._get_eval_fn(self, input_type)

    def _get_config(self, value):
        return {
            self.FIVR_5K: FIVR(version='5k'),
            self.FIVR_200K: FIVR(version='200k'),
            self.EVVE: EVVE(),
            self.SVD: SVD(),
        }[value]
