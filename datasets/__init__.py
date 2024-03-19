from typing import Dict

from datasets.base_text_detection_dataset import (  # noqa
    BaseTextDetectionDataset,
    BaseTextDetectionSample,
    BaseTextDetectionAnnotation)
from datasets.ICDAR2003 import ICDAR2003_dataset
from datasets.MSRA_TD500 import MSRA_TD500_dataset
from datasets.StreetViewText import SVT_dataset
from datasets.NEOCR import NEOCR_dataset

datasets: Dict[str, BaseTextDetectionDataset] = {
    'ICDAR2003': ICDAR2003_dataset,
    'MSRA_TD500': MSRA_TD500_dataset,
    'NEOCR': NEOCR_dataset,
    'StreetViewText': SVT_dataset
}
