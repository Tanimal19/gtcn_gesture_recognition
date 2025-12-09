from src.dataset_utils import DSequence, DAnnotation
from abc import ABC, abstractmethod
import torch


class AbstractModelRunner(ABC):
    @staticmethod
    @abstractmethod
    def predict_annotation(model, sequence: DSequence) -> DAnnotation:
        pass
