import os
import time
import random
from math import log2

_suffixes = ['bytes', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB', 'YiB']


def file_size(size):
    """
    Converting number of bytes to a human readable short form.

    Parameters
    ----------
    size: int
        A number represeting of file size.

    Returns
    -------
    A string represented number of bytes in short based degree of nubmbers.
    """
    order = int(log2(size) / 10) if size else 0

    return '{:.4g} {}'.format(size / (1 << (order * 10)), _suffixes[order])

def randomly_image_interval(digit_width=28, digit_count=1, image_width=28, spacing=(0, 0)):
    """
    Creating two generators of widths of image and spacing.
    A sequence of an image has a constant numbers
    A sequence of a spacing has a random numbers in a spacing range.

    Parameters
    ----------
    digit_width: int      Default: 28
        A width of one digit image.

    digit_count: int      Default: 1
        A count of digits in a sequences.

    image_width: int      Default: 28
        A width of the result sequence image. Using just for compatibility with "evenly_image_interval"

    spacing_range: tuple  Default: (0, 0)
        A (minimum, maximum) pair (tuple), representing the min and max spacing between digits.
        A unit should be a pixel.

    Returns
    -------
    A tuple of two sequences.
    First of them is a sequences of widths to resize a digit image, second is a sequence of widths of spacing.
    """
    return evenly_interval(digit_count, digit_width), random_interval(digit_count-1, spacing)


def evenly_image_interval(digit_width=28, digit_count=1, image_width=28, spacing=(0, 0)):
    """
    Creating two generators of widths of image and spacing.
    A purpose is an evenly placed images of digit and spacing between them
    and getting a width of the resulting image the same requirement.
    Generators will use in resize and spacing filters.

    Parameters
    ----------
    digit_width: int      Default: 28
        A width of one digit image.

    digit_count: int      Default: 1
        A count of digits in a sequences.

    image_width: int      Default: 28
        A width of the result sequence image.

    spacing_range: tuple  Default: (0, 0)
        A (minimum, maximum) pair (tuple), representing the min and max spacing between digits.
        A unit should be a pixel.

    Returns
    -------
    A tuple of two sequences.
    First of them is a sequences of widths to resize a digit image, second is a sequence of widths of spacing.
    """
    spacing_count = digit_count-1

    min_spacing, max_spacing = spacing
    # special case - can't changing spacing, change an image width
    if min_spacing == max_spacing:
        total_spacing = spacing_count * min_spacing
        image_width_rest = image_width - total_spacing
        result_width = int(image_width_rest/digit_count)

        return most_evenly_interval(digit_count, result_width, image_width_rest), evenly_interval(spacing_count, min_spacing)

    min_total_spacing = spacing_count * min_spacing
    max_total_spacing = spacing_count * (max_spacing-1)

    min_image_width = int((image_width - min_total_spacing)/digit_count)
    max_image_width = int((image_width - max_total_spacing)/digit_count)

    result_width = digit_width

    if min_image_width < digit_width:
        result_width = min_image_width
    elif max_image_width > digit_width:
        result_width = max_image_width

    spacing_rest = image_width - result_width * digit_count
    spacing_width = int(spacing_rest/spacing_count) if spacing_count > 0 else 0

    return evenly_interval(digit_count, result_width), most_evenly_interval(spacing_count, spacing_width, spacing_rest)


def evenly_interval(total_count, width):
    """
    Creating a generator of constant (evenly) values.

    Parameters
    ----------
    total_count: int
        Count of times of repeating.

    width: int
        A value repeated in the result generator

    Returns
    -------
    A generator of a constant value.
    """
    return (width for _ in range(total_count))


def most_evenly_interval(total_count, width, interval_rest):
    """
    Creating a generator of evenly equal values.
    A result width (interval_rest) might have a rest value from dividing total width and an image of digit width.
    Generator increases a value on 1 or staying it the same to fill all width.

    Parameters
    ----------
    total_count: int
        Count of times of repeating.

    width: int
        A value repeated in the result generator
    
    interval_rest: int
        A rest of width to place an image with minimum width.

    Returns
    -------
    A generator of a evenly placed values maximum differents is 1 for generated values.
    """
    def reduce_rest(rest, interval_rest=[interval_rest]):
        diff = interval_rest[0]-(rest*width)

        if diff > 0 and rest/diff % 2:
            interval_rest[0] -= width + 1
            return width + 1
        else:
            interval_rest[0] -= width
            return width

    return (reduce_rest(total_count-i) for i in range(total_count))

def random_interval(total_count, rng):
    """
    Creating a generator of a random number in the range.

    Parameters
    ----------
    total_count: int
        Count of times of repeating.

    range: tuple
        A value repeated in the result generator

    Returns
    -------
    A generator of a random number in the range.
    """
    return (random.randint(*rng) for _ in range(total_count))

def not_exists_file_name(file_name):
    """
    Checking a file using file name and changing it's base name if the file exists.
    A modification is a timestamp with brackets.
    Checking continue while the file with modified name exists or till 1000 times trying.

    Parameters
    ----------
    file_name: string
        A name of the file which needs to checking.

    Returns
    -------
    A modified file name or the same if the file doesn't exist.
    """
    new_file_name = file_name
    name, ext = os.path.splitext(file_name)
    
    counter = 0
    while os.path.exists(new_file_name) and counter < 1000:
        new_file_name = "{} ({}){}".format(name, int(time.time()), ext)

        counter += 1

    return new_file_name
