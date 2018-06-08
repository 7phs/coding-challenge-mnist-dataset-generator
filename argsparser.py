import argparse


class SpacingAction(argparse.Action):
    """
    A class is extending parsing of a spacing range argument value.
    A string value will convert to a tuple of the minimum and maximum width of spacing.
    """

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(SpacingAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        parts = values.split(",")
        if len(parts) == 1:
            values = (int(parts), int(parts))
        elif len(parts) > 1:
            values = (int(parts[0]), int(parts[1]))

        setattr(namespace, self.dest, values)


def parser():
    """
    Create a parser of console arguments of a generator tool.

    Requirements arguments:
        digits:
        A string each digit characters which sequenced transformed to an image using MNIST images.

    Optiononal arguments:
        -o | --output:   Default: mnist_numbers_sequence.png
            A name of a result PNG image.

        -d | --data_directory:
            A custom path to cache MNIST datafile.

        -w | --image_width:
            A width of a result image.

        -s | --spacing:   Default: 0,0
            A range of spacing between digit images separated a comma.
            Format: minimum,maximum
            Example:
                3
                2,100

        -e | --evenly    Default: off
            An evenly spaced of spacing against a randomly choosen in the spacing range.

        -f | --filters:
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

    Returns
    -------
    An object of ArgumentParser which possible manual executes parsing arguments, storing a result of parsing
    and getting an argument values as a property.
    """
    # TODO: add buring and wraping
    parser = argparse.ArgumentParser(
        description='Generate an image of numbers sequence')
    parser.add_argument('-o', '--output', metavar='image_name.png', default='mnist_numbers_sequence.png',
                        help='a name of a result PNG file. Default: mnist_numbers_sequence.png', )
    parser.add_argument('-d', '--data_directory', type=str,
                        help='a directory stored downloaded MNIST data files')
    parser.add_argument('-w', '--image_width', type=str,
                        help='a width of the result image')
    parser.add_argument('-s', '--spacing', metavar='min,max', type=str, action=SpacingAction,
                        help='a spacing range min,max. Default: 0,0')
    parser.add_argument('-e', '--evenly', action='store_true',
                        help='an evenly placed of spacing against a default randomly choosen in the spacing range.')
    parser.add_argument('-f', '--filters', metavar='filter1,filter2', type=str,
                        help='additional filters applyed on digit images. Supported filters: "blur" and "distort" ')
    parser.add_argument('digits', help='a numbers sequence')

    return parser
