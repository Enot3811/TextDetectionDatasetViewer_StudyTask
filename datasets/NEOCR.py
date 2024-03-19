"""NEOCR dataset classes."""


from pathlib import Path
from typing import List, Union
import xml.etree.ElementTree as ET

from datasets import (
    BaseTextDetectionDataset,
    BaseTextDetectionSample,
    BaseTextDetectionAnnotation)


class NEOCR_annotation(BaseTextDetectionAnnotation):
    pass


class NEOCR_sample(BaseTextDetectionSample):
    pass


class NEOCR_dataset(BaseTextDetectionDataset):
    def __init__(self, dset_folder: Union[Path, str]) -> None:
        super().__init__(dset_folder)

        annots_dir = self.dset_folder / 'Annotations'
        img_dir = self.dset_folder / 'Images'
        annots_files = annots_dir.glob('*.xml')

        self._subsets['train'] = [
            self.read_annotation_file(annots_file, img_dir)
            for annots_file in annots_files]

    def read_annotation_file(
        self, annot_pth: Path, img_dir: Path
    ) -> NEOCR_sample:
        """Read an annotation file of NEOCR dataset.

        Make an sample's annotation based on the read file.

        Parameters
        ----------
        annot_pth : Path
            A path to the file.
        img_dir : Path
            A path to an image directory.

        Returns
        -------
        NEOCR_sample
            The read sample annotation.
        """
        tree = ET.parse(annot_pth)
        root = tree.getroot()

        img_pth = img_dir / root[0].text
        
        annots: List[NEOCR_annotation] = []
        for image_annot in root.findall('object'):
            word = image_annot[0].text
            language = image_annot[7][1].text
            bbox = image_annot[9]

            x1 = int(bbox[1][0].text)
            y1 = int(bbox[1][1].text)
            x2 = int(bbox[3][0].text)
            y2 = int(bbox[3][1].text)

            annots.append(NEOCR_annotation(x1, y1, x2, y2, language, word))
        return NEOCR_sample(img_pth, annots)
