import numpy as np
import random
from skimage.transform import resize as rz
from scipy.ndimage.filters import gaussian_filter


def invert(max_v):
    """
    A filter is inverting each pixel on an interval [0, max_v]

    Parameters
    ----------
    max_v: int
        A maximum value of an array. Each array element value will subtract from the parameter.

    Returns
    --------
    A function will apply on ndarray.
    """
    return np.vectorize(lambda x: np.uint8(max_v-x))


def normalize(max_v):
    """
    A filter is normalize each float value on an interval [0.0, 1.0]

    Parameters
    ----------
    max_v: int
        A maximum value of an array. Each array element value will divide on the parameter.

    Returns
    --------
    A function will apply on ndarray.
    """
    return np.vectorize(lambda x: x/np.float32(max_v))


def resize_seq(digit_width_seq, default=0):
    """
    A filter is resizing ndarray on X-axis.
    A new width to resize array will get from a sequence to getting evenly sized images.
    Resizing will not executing in 2 cases. First, the new width is zero or none.
    Second, the new width is equal exists.

    Parameters
    ----------
    digit_width_seq: generator of ints
        A sequence of new widths for an image. A sequence makes possible getting evenly width of each sequence image.
        New width would setting to default parameter if the sequence finished.

    default: int   Default: 0
        A default value of a finished sequence.

    Returns
    --------
    A function will apply on ndarray.
    """
    def resize_image(img):
        digit_width = next(digit_width_seq, default)
        if digit_width and digit_width != default:
            return rz(img, (img.shape[0], digit_width),  mode='wrap')
        else:
            return img

    return resize_image

def resize(width):
    """
    A filter is resizing ndarray on X-axis.

    Parameters
    ----------
    width: int
        A target width of the image

    Returns
    -------
    A function will apply on ndarray.
    """
    def resize_image(img):
        return rz(img, (img.shape[0], width),  mode='wrap')

    return resize_image
    

def spacing_seq(spacing_width_seq, max_v, default=None):
    """
    A filter is extending an array by empty, spacing array on the right side of X-axis.
    A width of extending will get from a sequence to getting evenly sized images.
    Extending will not apply if getting zero. It helps to skip extending on the last image.

    Parameters
    ----------
    spacing_width_seq: generator of ints
        A sequence of new spacing widths for an image. A sequence makes possible the acquisition of evenly the width of each sequence image.
        New width would be set to default parameter if the sequence finished.

    max_v: int
        A parameter for an invert filter.
        An invert filter will apply a new ndarray to get expected array values.

    default: int Default: None
        A default value of a finished sequence.
        Use a None or 0 to stop extension an image if sequence finished.

    Returns
    --------
    A function will apply on ndarray.
    """
    def add_spacing(img):
        space_width = next(spacing_width_seq, default)
        if space_width:
            space = np.zeros(shape=(28, space_width), dtype=np.float32)
            space.fill(1.0)
            return np.concatenate((img, space), axis=1)
        else:
            return img

    return add_spacing


def blur(v=2):
    """
    A filter is blurring an image.

    Parameters
    ----------
    v: int   Default: 2
        A coefficient of blurring. Use larger value to increase a blurring.

    Returns
    --------
    A function will apply on ndarray.
    """
    def blur_image(img):
        return gaussian_filter(img, sigma=v)

    return blur_image


def distort(alpha):
    """
    A filter is distorting an image using random horizontal rolling each line.

    Parameters
    ----------
    alpha: int
        A coefficient to generate a random value of rolling
        each X-axis row of ndarray represented an image.
        Recommended value from 5 to 20.

    Returns
    --------
    A function will apply on ndarray.
    """
    def distort_image(img):
        A = img.shape[0] / 1.5

        def shift(x): return A * (random.randrange(0, alpha)/100)

        for i in range(img.shape[0]):
            img[i, :] = np.roll(img[i, :], int(shift(i)))

        return img

    return distort_image
