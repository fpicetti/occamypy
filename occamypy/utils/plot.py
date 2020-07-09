import numpy as np
import imageio
from occamypy.vector import Vector


def vector2gif(data, filename, transpose=False, clip=100, frame_axis=-1, frame_interval=.25):
    """
    Save a 3D vector to an animated GIF file.
    
    :param data: vector - must have 3 dimensions
    :param filename: `str` - path to file to be written with extension
    :param transpose: `bool` - transpose the image axis [False]. Notice that sepvector have the axis swapped w.r.t. numpy
    :param clip: `float` - percentile for clipping the data
    :param frame_axis: `int` - frame axis of data [-1]
    :param frame_interval: `float`- time between frames [0.25]

    """
    assert isinstance(data, Vector), "Data has to be a vector instance"
    assert data.ndims == 3, "Data has to be a 3D vector"
    
    clip_val = np.percentile(np.absolute(data), clip)
    data = np.clip(data, -clip_val, clip_val)
    if frame_axis > 0:  # need to bring the dim. to first dim
        data = np.swapaxes(data, frame_axis, 0)
    if transpose:
        frames = [data[_].T for _ in range(data.shape[0])]
    else:
        frames = [data[_] for _ in range(data.shape[0])]

    imageio.mimsave(filename, frames, 'GIF', **{'duration': frame_interval})

