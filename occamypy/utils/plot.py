from typing import Tuple

import numpy as np
from imageio import mimsave


def float2png(in_content: np.ndarray) -> np.ndarray:
    """
    Convert a np.ndarray dynamics to [0,255] uint8
    
    Args:
        in_content: float image to be converted

    Returns:
        image in the png dynamics
    """
    in_min = np.min(in_content)
    in_max = np.max(in_content)
    in_content = 255 * (in_content - in_min) / (in_max - in_min)  # x in [0,255]
    return in_content.astype(np.uint8)


def vector2gif(in_content: np.ndarray, filename: str, transpose: bool = False, clip: float = 100., frame_axis=-1,
               fps: int = 25) -> None:
    """
    Save a 3D vector to an animated GIF file.
    
    Args:
        in_content: 3D numpy.array
        filename: path to file to be written with extension
        transpose: whether to transpose the image axis [False]. Notice that sepvector have the axis swapped w.r.t. numpy
        clip: percentile for clipping the data
        frame_axis: frame axis of data [-1]
        fps: number of frames per second [25]
    """
    if in_content.ndim != 3:
        raise ValueError("in_content has to be a 3D vector")
    
    if clip != 100.:
        clip_val = np.percentile(np.absolute(in_content), clip)
        in_content = np.clip(in_content, -clip_val, clip_val)
    
    in_content = float2png(in_content)
    
    if frame_axis != 0:  # need to bring the dim. to first dim
        in_content = np.swapaxes(in_content, frame_axis, 0)
    if transpose:
        frames = [in_content[_].T for _ in range(in_content.shape[0])]
    else:
        frames = [in_content[_] for _ in range(in_content.shape[0])]
    
    mimsave(filename, frames, **{'fps': fps})


def clim(in_content: np.ndarray, ratio: float = 95) -> Tuple[float, float]:
    """
    Compute the clim tuple for plotting an array
    
    Args:
        in_content: array to be analyzed
        ratio: percentile value for clipping

    Returns:
        tuple (-c, c) of clipping values
    """
    c = np.percentile(np.absolute(in_content), ratio)
    return -c, c
