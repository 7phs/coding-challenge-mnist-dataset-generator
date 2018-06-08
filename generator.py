import imageio
import re
import numpy as np

if __name__.find('.')<0:
    import filters
    import argsparser
    import helper
    import mnistdata
else:
    from . import filters
    from . import argsparser
    from . import mnistdata
    from . import helper

GENERATOR_MINIMUM_IMAGE_WIDTH = 10

def default_parameters(digit_width, digits_len, spacing_range, image_width):
    """
    Checking generator parameters and set default values if getting None or empty.

    Parameters
    ----------
    digit_width: int
        The standard width of an image stored in MNIST DB.

    digits_len: int
        A count of a digit of the generated sequence.

    spacing_range: tuple
        A (minimum, maximum) pair (tuple), representing the min and max spacing between digits.
        A unit should be a pixel.

    image_width: int
        specifies the width of the image in pixels.

    Returns
    --------
    A tuple of fixed or the same spacing_range and image_width parameters.
    """
    # init default spacing
    if not spacing_range:
        spacing_range = (0, 0)
    # default image width based digits count
    if not image_width or not int(image_width):
        image_width = digits_len * digit_width + (digits_len - 1) * \
            spacing_range[0]
    else:
        image_width = int(image_width)

    return (spacing_range, image_width)

def check_parameters(digits_len, spacing_range, image_width):
    """
    Checking parameters using requirements:
    - a sequence has to contain at least one element;
    - spacing range has to order;
    - spacing range has to have zero or positive numbers;
    - image_width has to be a positive;
    - minimum possible digital image width has to great than a minimum.

    Parameters
    ----------
    digits_len: int
        A count of a digit of the generated sequence.

    spacing_range: tuple
        A (minimum, maximum) pair (tuple), representing the min and max spacing between digits.
        A unit should be a pixel.

    image_width: int
        specifies the width of the image in pixels.

    Raises
    ------
    An exception containing a string described all of the error of checking parameters.
    """
    errors = []

    # empty sequence
    if digits_len == 0:
        raise Exception("nothing to generate empty serquence")

    spacing_count = digits_len-1

    # wrong order
    minimum, maximum = spacing_range
    if minimum > maximum:
        errors.append("spacing range {} has invalid values: minimum={} greater than maximum={}".format(
            spacing_range, minimum, maximum))
        minimum, maximum = maximum, minimum

    # negative spacing
    if minimum < 0 or maximum < 0:
        errors.append(
            "spacing range {} has negative numbers".format(spacing_range))

    # negative image width
    if image_width<=0:
        errors.append(
            "image width {} is negative numbers or zero".format(image_width))
    else:
        # minimum digit width
        image_rest_width = image_width - spacing_count * minimum
        digit_width = int(image_rest_width/digits_len)
        if digit_width < GENERATOR_MINIMUM_IMAGE_WIDTH:
            errors.append("image width {} and spacing range {} produces a digit image width {} less than minimum requirement {}. Change that parameters to prevent an error".format(
                image_width, spacing_range, digit_width, GENERATOR_MINIMUM_IMAGE_WIDTH))

    if errors:
        raise Exception("; ".join(errors))

def get_filters(digit_width, digit_max_value, digits_len, spacing_range, image_width,
                evenly=False, fltrs=None):
    """
    Getting complete list of filters to process a digit images: resizing, spacing, etc.

    Parameters
    ----------
    digit_width: int
        The standard width of an image stored in MNIST DB.

    digit_max_value: int
        The max (white) value of the image array.

    digits_len: int
        A count of a digit of the generated sequence.

    spacing_range: tuple
        A (minimum, maximum) pair (tuple), representing the min and max spacing between digits.
        A unit should be a pixel.

    evenly: boolean    Default: False
        A mode of generating an image.
        If False - Randomly choosing a spacing in the spacing_range.
        If True - evenly interval for each image and spacing.

    fltrs: list of functions
        A list-like containing functions. Each of them will apply on a digit image and modify it
        before adding to sequence.

    Return
    ------
    A list-like containing filter functions.
    The result has to contain a default filters like as invert, resize and spacing
    and might be extending a custom list of filters.
    Each of them will apply on a digit image and modify it before adding to sequence.
    """
    # image parameters
    digit_count = digits_len
    digit_width = digit_width
    # default filter - invert
    processing_filters = [filters.invert(digit_max_value), filters.normalize(digit_max_value)]
    if evenly and fltrs:
        processing_filters += fltrs
    # calc image and spacing parameters
    creating_interval = helper.randomly_image_interval if not evenly else helper.evenly_image_interval

    digit_width_seq, spacing_width_seq = creating_interval(
        digit_width=digit_width,
        digit_count=digit_count,
        image_width=image_width,
        spacing=spacing_range)
    # add default post process filters
    processing_filters.append(filters.resize_seq(
        digit_width_seq, default=digit_width))
    processing_filters.append(filters.spacing_seq(
        spacing_width_seq, digit_max_value))

    postprocessing_filters = []
    if not evenly:
        if fltrs:
            postprocessing_filters += fltrs
        postprocessing_filters.append(filters.resize(image_width))

    return processing_filters, postprocessing_filters

def generate_numbers_sequence(digits, spacing_range, image_width,
                              data_home=None,
                              images=None,
                              evenly=False,
                              fltrs=None):
    """
    Generate an image that contains the sequence of given numbers, spaced evenly or
    randomly using a uniform distribution.

    Parameters
    ----------
    digits: list of ints
        A list-like containing the numerical values of the digits from which the sequence
        will be generated (for example [3, 5, 0]).

    spacing_range: tuple
        A (minimum, maximum) pair (tuple), representing the min and max spacing between digits.
        A unit should be a pixel.

    image_width: int
        specifies the width of the image in pixels.

    data_home: str  Default: None
        A custom path of storing MNIST datafiles.
    
    images: object
        A custom MNIST image db to prevent using default DB of a mnistdata module.
        Ex. it useful for testing.

    evenly: boolean    Default: False
        A mode of generating an image.
        If False - Randomly choosing a spacing in the spacing_range.
        If True - evenly interval for each image and spacing.
    
    fltrs: list of functions
        A list-like containing functions. Each of them will apply on a digit image and modify it
        before adding to sequence.

    Returns
    -------
    The image containing the sequence of numbers. The image is representing
    as floating point 32bits numpy arrays with a scale ranging from 0 (black) to 1 (white).
    """
    # get MNIST images db
    if images is None:
        images = mnistdata.get_images(data_home=data_home)
    # set default values
    spacing_range, image_width = default_parameters(images.digit_width(), len(digits), spacing_range, image_width)

    check_parameters(len(digits), spacing_range, image_width)
    # get image filters
    processing_filters, postprocessing_filters = get_filters(
        images.digit_width(), images.max_value(),
        len(digits), spacing_range, image_width,
        evenly=evenly,
        fltrs=fltrs)
    # prepare init array
    result_img = np.zeros(shape=(28, 0), dtype=np.float32)
    # add all digits into image, but slicing it to get a on digit numbers
    for digit in [d%10 for d in digits]:
        img = images[digit]

        for fltr in processing_filters:
            img = fltr(img)

        result_img = np.concatenate((result_img, img), axis=1)

    # apply post processing filters
    for fltr in postprocessing_filters:
        result_img = fltr(result_img)

    return result_img


if __name__ == '__main__':
    """
    A tool generating a PNG image from a digit sequence using random prepared handwritten symbols of MNIST database.

    Example:
        python generator.py -o first_image.png -w 360 -s 0,20 -f distort 498127864687234

    Requirements arguments:
        digits
        A string each digit characters which sequenced transformed to an image using MNIST images.

    Optiononal arguments:
        -o | --output   Default: mnist_numbers_sequence.png
            A name of a result PNG image.

        -d | --data_directory
            A custom path to cache MNIST datafile.

        -w | --image_width
            A width of a result image.

        -s | --spacing   Default: 0,0
            A range of spacing between digit images separated a comma.
            Format: minimum,maximum
            Example:
                3
                2,100

        -e | --evenly    Default: off
            An evenly spaced of spacing against a default randomly choosen in the spacing range.

        -f | --filters
            A list of additional filters applied each digit image.
            Supported filters:
                "blur" - a little bit blurring an image
                "distort" - make a random horizontal moving each line of an image.
            Filter applied by order of value.
            Format: filter1,filter2
            Example:
                blur
                distort,blur
                blur,distort
    """

    def parse_filters(filters_str):
        """
        Parsing a string to a list-like which contains filter functions to modify digit image.

        Parameters
        ----------
        filters_str: str
            A string constaing list of name of filters separated comma.

        Returns
        -------
        A list-like containing filter functions.
        """
        fltrs = []
        for part in str(filters_str).lower().split(","):
            if part=="blur":
                fltrs.append(filters.blur(1))
            elif part=="distort":
                fltrs.append(filters.distort(18))

        return fltrs

    # parse arguments
    args = argsparser.parser().parse_args()
    # generate a dataset based numbers sequence of digits
    # try:
    print("Generate an image")
    dataset = generate_numbers_sequence(
        [int(digit) for digit in args.digits],
        args.spacing,
        args.image_width,
        data_home=args.data_directory,
        evenly=args.evenly,
        fltrs=parse_filters(args.filters))
    # except Exception as e:
    #     print("failed to generate an image array: ", e)
    #     exit(-1)
    # store a dataset as a PNG images
    try:
        image_file_name = helper.not_exists_file_name(args.output)

        print("Store an image into '{}'".format(image_file_name))

        imageio.imwrite(image_file_name,
            np.vectorize(lambda x: np.uint8(255 * x))(dataset))
    except Exception as e:
        print("failed to store an image based a generated array", e)
        exit(-1)
