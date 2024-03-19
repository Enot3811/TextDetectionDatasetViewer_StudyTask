"""The module contains a base dataset classes for object detection task.

Each dataset consists of a dataset, a sample and an annotation classes.
"""


from utils.data_utils.datasets import (
    BaseObjectDetectionDataset,
    BaseObjectDetectionSample,
    BaseObjectDetectionAnnotation)


class BaseTextDetectionAnnotation(BaseObjectDetectionAnnotation):
    def __init__(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        language: str = 'unlabeled',
        text: str = ''
    ) -> None:
        super().__init__(x1, y1, x2, y2, language)
        self.text = text


class BaseTextDetectionSample(BaseObjectDetectionSample):
    pass


class BaseTextDetectionDataset(BaseObjectDetectionDataset):
    pass
