"""A module that contain functions for working with images."""


from pathlib import Path
from typing import Tuple, Union, List

from numpy.typing import NDArray
import cv2


IntBbox = Tuple[int, int, int, int]
FloatBbox = Tuple[float, float, float, float]
Bbox = Union[IntBbox, FloatBbox]


def read_image(path: Union[Path, str], grayscale: bool = False) -> NDArray:
    """Read image to numpy array.

    Parameters
    ----------
    path : Union[Path, str]
        Path to image file
    grayscale : bool, optional
        Whether read image in grayscale, by default False

    Returns
    -------
    NDArray
        Array containing read image.

    Raises
    ------
    FileNotFoundError
        Did not find image.
    ValueError
        Image reading is not correct.
    """
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Did not find image {path}.')
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise ValueError('Image reading is not correct.')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def resize_image(image: NDArray, new_size: Tuple[int, int]) -> NDArray:
    """Resize image to given size.

    Parameters
    ----------
    image : NDArray
        Image to resize.
    new_size : Tuple[int, int]
        Tuple containing new image size.

    Returns
    -------
    NDArray
        Resized image
    """
    return cv2.resize(
        image, new_size, None, None, None, interpolation=cv2.INTER_LINEAR)


def save_image(img: NDArray, path: Union[Path, str]) -> None:
    """Save a given image to a defined path.

    Parameters
    ----------
    img : NDArray
        The saving image.
    path : Union[Path, str]
        The save path.

    Raises
    ------
    RuntimeError
        Could not save image.
    """
    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError('Could not save image.')


def draw_bounding_boxes(
    image: NDArray,
    bboxes: List[Bbox],
    class_labels: List[Union[str, int, float]] = None,
    confidences: List[float] = None,
    line_width: int = 1,
    color: Tuple[int, int, int] = (255, 255, 255),
    exclude_classes: List[Union[str, int, float]] = None
) -> NDArray:
    """Draw bounding boxes and corresponding labels on a given image.

    Parameters
    ----------
    image : NDArray
        The given image with shape `(h, w, c)`.
    bboxes : List[Bbox]
        The bounding boxes with shape `(n_boxes, 4)` in `xyxy` format.
    class_labels : List, optional
        Bounding boxes' labels. By default is None.
    exclude_classes : List[str, int, float]
        Classes which bounding boxes won't be showed. By default is None.
    confidences : List, optional
        Bounding boxes' confidences. By default is None.
    line_width : int, optional
        A width of the bounding boxes' lines. By default is 1.
    color : Tuple[int, int, int], optional
        A color of the bounding boxes' lines in RGB.
        By default is `(255, 255, 255)`.

    Returns
    -------
    NDArray
        The image with drawn bounding boxes.
    """
    image = image.copy()
    if exclude_classes is None:
        exclude_classes = []

    for i, bbox in enumerate(bboxes):
        # Check if exclude
        if class_labels is not None and class_labels[i] in exclude_classes:
            continue

        # Draw bbox
        bbox = list(map(int, bbox))  # convert float bbox to int if needed
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2),
                      color=color, thickness=line_width)
        
        # Put text if needed
        if class_labels is not None:
            put_text = f'cls: {class_labels[i]} '
        else:
            put_text = ''
        if confidences is not None:
            put_text += 'conf: {:.2f}'.format(confidences[i])
        if put_text != '':
            cv2.putText(image, put_text, (x1, y1 - 2), 0, 0.3,
                        color, 1)
    return image
