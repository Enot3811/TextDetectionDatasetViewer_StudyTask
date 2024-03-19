"""ICDAR 2003 dataset classes."""


from pathlib import Path
from typing import List, Union
import xml.etree.ElementTree as ET

from datasets import (
    BaseTextDetectionDataset,
    BaseTextDetectionSample,
    BaseTextDetectionAnnotation)


class ICDAR2003_annotation(BaseTextDetectionAnnotation):

    def __init__(
        self,
        x: Union[int, float],
        y: Union[int, float],
        w: Union[int, float],
        h: Union[int, float],
        text: str
    ) -> None:
        x1 = int(x)
        y1 = int(y)
        x2 = x1 + int(w)
        y2 = y1 + int(h)
        language = 'english'
        super().__init__(x1, y1, x2, y2, language, text)


class ICDAR2003_sample(BaseTextDetectionSample):
    pass


class ICDAR2003_dataset(BaseTextDetectionDataset):
    def __init__(self, dset_folder: Union[Path, str]) -> None:
        super().__init__(dset_folder)

        train_dir = self.dset_folder / 'SceneTrialTrain'
        test_dir = self.dset_folder / 'SceneTrialTest'
        sample_dir = self.dset_folder / 'SceneTrialSample'

        self._subsets['sample'] = self.read_set(sample_dir)
        self._subsets['train'] = self.read_set(train_dir)
        self._subsets['test'] = self.read_set(test_dir)

    def read_set(self, set_dir: Path) -> List[ICDAR2003_sample]:
        """Read a directory with a set, generate a list of samples.

        Parameters
        ----------
        set_dir : Path
            The set directory path.

        Returns
        -------
        List[ICDAR2003_sample]
            The list of set's samples.
        """
        tree = ET.parse(set_dir / 'words.xml')
        root = tree.getroot()

        samples: List[ICDAR2003_sample] = []
        for image_annots in root:
            img_pth = set_dir / image_annots[0].text
            bboxes = image_annots[2]

            annots: List[ICDAR2003_annotation] = []
            for bbox in bboxes:
                x = int(bbox.attrib['x'].split('.')[0])
                y = int(bbox.attrib['y'].split('.')[0])
                w = int(bbox.attrib['width'].split('.')[0])
                h = int(bbox.attrib['height'].split('.')[0])
                word = bbox[0].text
                annots.append(ICDAR2003_annotation(x, y, w, h, word))

            samples.append(ICDAR2003_sample(img_pth, annots))
        return samples
