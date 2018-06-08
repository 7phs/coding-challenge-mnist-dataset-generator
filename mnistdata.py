import io
import os
import struct
import sys
import random
import numpy as np
from collections import defaultdict

if __name__.find('.')<0:
    import mnistdownloader
else:
    from . import mnistdownloader

DATAHOME_ENV_NAME = 'GENERATOR_NUMBERS_SEQ_MNIST_DIR'
DATAHOME_DEFAULT_PATH = 'generator_numbers_seq_mnist'

MNIST_READ_SIZE = 64 * 1024

MNIST_DEFAULT_IMAGE_WIDTH = 28
MNIST_DEFAULT_IMAGE_HEIGHT = 28

mnist_labeles = None
mnist_images = None


def get_data_file_path(file_name, data_home=None):
    """
    Getting a full storages path of DB file.
    Datafile directry is configurable using an environment parameter GENERATOR_NUMBERS_SEQ_MNIST_DIR.
    Default path is $HOME/generator_numbers_seq_mnist

    Parameters
    ----------
    file_name: str
        Name of a DB file which is storing or loading.

    data_home: str      Default: None
        A custom path was storing DB files.

    Returns
    -------
    A full local path of a file.
    """
    if data_home is None:
        data_home = os.environ.get(
            DATAHOME_ENV_NAME, os.path.join('~', DATAHOME_DEFAULT_PATH))

    data_home = os.path.expanduser(data_home)

    if not os.path.exists(data_home):
        os.makedirs(data_home)

    return os.path.join(data_home, file_name)


class MNISTDataFile:
    """
    Base class implementing a general operation for a datafile: fetching from a remote and read a header.
    """

    def __init__(self,
                 title="unknown",
                 downloader=mnistdownloader.HttpDownloader,
                 file_name=None,
                 header_magic_numer=2000):
        """
        Parameters
        ----------
        title: str
            A title of DB helping debug of operation on datafile.

        downloader: object   Default: mnistdownloader.HttpDownloader
            An object is implementing of getting a datafile from another (remote or local)
            resource.

        file_name: str
            A name of a datafile.

        header_magic_number: int
            The first value of a header. It helps to check an opened datafile.
        """
        self.file_name = file_name
        self.file_path = None
        self.file_size = 0
        self.title = title
        self.header_magic_numer = header_magic_numer
        self.header_format = ">II"
        self.downloader = downloader
        self.data_file = None
        self.reader = None
        self.fetcher = None
        self.record_count = 0

    def fetch(self, data_home=None):
        """
        Downloading of a datafile from another (remote or local) resource.

        Parameters
        ----------
        data_home: str      Default: None
            Custom path storing DB files.

        Raises
        -------
        An exception related http errors or errors of storing a downloadable datafile.
        """
        # checking a data file
        self.file_path = get_data_file_path(self.file_name, data_home=data_home)
        if os.path.exists(self.file_path):
            return
        # storing a data file to a local file
        try:
            self.fetcher = self.downloader(self.file_name)

            print("Download a MNIST ", self.title,
                  " file: ", self.fetcher.remote_file_name, self.fetcher.file_size)

            with open(self.file_path, 'wb') as data_file:
                writer = io.BufferedWriter(data_file)

                data = self.fetcher.read()
                while data:
                    writer.write(data)
                    data = self.fetcher.read()

                writer.flush()
            # checking a real downloaded size and expected
            self.fetcher.close()
            self.fetcher.check_downloaded_size()

        except Exception as e:
            # TODO: process exception, but only espessialy exeption
            print("HTTPError: ", e)

            if os.path.exists(self.file_path):
                os.remove(self.file_path)

            raise e

    def read(self, data_home=None):
        """
        Opening a datafile and reading a general header to start processing it.

        Parameters
        ----------
        data_home: str      Default: None
            Custom path storing DB files.

        Raises
        ------
        An exception related errors of reading a downloadable datafile or checking an invalid header of it.
        """
        if not self.data_file is None:
            return

        self.file_path = get_data_file_path(self.file_name, data_home=data_home)
        self.data_file = open(self.file_path, "rb")

        self.reader = io.BufferedReader(self.data_file)

        header = self.read_header()
        if header[0] != self.header_magic_numer:
            # TODO: specify an exception
            raise Exception("error header")

        self.record_count = header[1]

    def read_header(self):
        """
        Read a general header of datafile and unpack it.

        Returns
        -------
        A tuple of header values: magic number and count of records.
        """
        return struct.unpack(self.header_format, self.reader.read(8))

    def check_content(self, expected_size):
        """
        Checking a value based on header value of count of records and exists size of datafile.

        Parameters
        ----------
        expected_size: int

        Raises
        ------
        An exception of unexpected header value of count of records.
        """
        total_size = os.path.getsize(self.file_path)
        if total_size!=expected_size:
            raise Exception("broken content: file size is {}, but expected is {}".format(total_size, expected_size))

    def close(self):
        """
        Closing all opened resources such as a buffered reader and a data file.
        """
        if not self.data_file is None:
            self.data_file.close()
            self.reader.close()
            self.data_file = None
            self.reader = None


class MNISTLabelsFile(MNISTDataFile):
    """
    A class of labels DB implementing a specific operations on a labels datafile.
    """

    def __init__(self, downloader=mnistdownloader.HttpDownloader):
        """
        Parameters
        ----------
        downloader: object   Default: mnistdownloader.HttpDownloader
            An object is implementing of getting a datafile from another (remote or local)
            resource.
        """
        super().__init__(
            title="labels",
            file_name="train-labels-idx1-ubyte",
            downloader=downloader,
            header_magic_numer=2049)

        self.indexes = defaultdict(lambda: [])

        random.seed()

    def read(self, data_home=None):
        """
        Opening a labels datafile and reading all of the data to memory.

        Parameters
        ----------
        data_home: str      Default: None
            Custom path storing DB files.

        Raises
        ------
        An exception related unexpected count of records different than a header parameter.
        """
        # open a file and read a general header
        super().read(data_home=data_home)
        super().check_content(8+self.record_count)
        # read labels
        index = 0
        buffer = self.reader.read(MNIST_READ_SIZE)
        record_count = 0
        while buffer:
            record_count+=len(buffer)

            for ch in buffer:
                self.indexes[ch].append(index)
                index += 1

            buffer = self.reader.read(MNIST_READ_SIZE)
        # close a file
        self.close()
        # check read records count
        if record_count!=self.record_count:
            raise Exception("read {} records, but expected is {}".format(record_count, self.record_count))

    def __getitem__(self, key):
        """
        Getting an index of one of the handwritten image of a digit.
        An index will select randomly from a list of all stored image of a requested digit.

        Parameters
        ----------
        key: int
            A digit from 0 to 9.

        Returns
        -------
        An index of handritten image of digit.
        """
        if not key in self.indexes:
            return -1

        index = random.randrange(len(self.indexes[key]))

        return self.indexes[key][index]


class MNISTImagesFile(MNISTDataFile):
    """
    A class of labels DB implementing a specific operations on a images datafile.
    """

    def __init__(self, labels, downloader=mnistdownloader.HttpDownloader):
        """
        Parameters
        ----------
        labels: object
            A DB of labeles containing offsets of a digit.

        downloader: object   Default: mnistdownloader.HttpDownloader
            An object is implementing of getting a datafile from another (remote or local)
            resource.
        """
        super().__init__(
            title="images",
            file_name="train-images-idx3-ubyte",
            downloader=downloader,
            header_magic_numer=2051)

        self.image_width = MNIST_DEFAULT_IMAGE_WIDTH
        self.image_height = MNIST_DEFAULT_IMAGE_HEIGHT
        self.image_header_format = ">II"
        self.start_offset = 0
        self.labels = labels

        self.__calc_record_offset()

    def __calc_record_offset(self):
        """
        Calculating an image offset or image array size using to calculate a file offset.
        """
        self.image_offset = self.image_width * self.image_height

    def digit_width(self):
        """
        Returns
        -------
        A width of one image of digit.
        """
        return self.image_width

    def max_value(self):
        """
        Returns
        -------
        A constant maximum value of an image array.
        """
        return 255

    def read(self, data_home=None):
        """
        Opening an images datafile and reading just a header.
        A datafile will opened to get an image data through all working time.

        Parameters
        ----------
        data_home: str      Default: None
            Custom path storing DB files.

        Raises
        ------
        An exception related unexpected count of records different than a header parameter.
        """
        # open a file and read a general header
        super().read(data_home=data_home)
        # read image specific header
        self.image_width, self.image_height = self.read_image_header()
        super().check_content(16+self.record_count * self.image_width * self.image_height)
        # store start index
        self.__calc_record_offset()
        self.start_offset = self.reader.tell()

    def read_image_header(self):
        """
        Read a specific header of images datafile and unpack it.

        Returns
        -------
        A tuple of header values: image width and height.
        """
        return struct.unpack(self.image_header_format, self.reader.read(8))

    def __getitem__(self, key):
        """
        Getting an image data.
        An offset of a datafile calculating on an index from labels db.
        Data reading from the datafile repeated each time.

        Parameters
        ----------
        key: int
            A digit from 0 to 9.

        Returns
        -------
        A numpy 2D array is containing uint8 elements.

        Raises
        ------
        An exceptions related file operations or getting unknown digits.
        """
        index = self.labels[key]
        if index < 0:
            # TODO: normal exceptions
            raise Exception("Unknown key")
        # find an image position and read it
        self.reader.seek(self.start_offset + index * self.image_offset)
        data = self.reader.read(self.image_offset)
        if not data:
            # TODO: normal exceptions
            raise Exception("Error read")

        data = np.array(list(data), dtype=np.uint8)

        return np.reshape(data, (-1, self.image_width))


def get_images(data_home=None):
    """
    Initializing images and labels DB and storing in a module variable.
    It is like as a singleton.

    Parameters
    ----------
    data_home: str      Default: None
        A custom path was storing DB files.

    Returns
    -------
    An DB objects containing handwritten images of digit.
    """
    global mnist_labeles, mnist_images

    if mnist_labeles is None:
        mnist_labeles = MNISTLabelsFile()

    if mnist_images is None:
        mnist_images = MNISTImagesFile(mnist_labeles)

    for o in [mnist_labeles, mnist_images]:
        o.fetch(data_home=data_home)
        o.read(data_home=data_home)

    return mnist_images


def GenerateTestData(target_dir, width, height, count, without_content=False):
    """
    Generating two datafiles and storing them into a target directory.
    Generated files are including valid headers.
    Contents of the files are just zero values elements excepts the first is equals a value of the generated digit.
    It is useful to check loaded images just sum of all array elements and comparing it with a requested digit.

    Parameters
    ----------
    target_dir: str
        A target directory to store testing datafiles.
        Default directory is current working directory.

    width: int
        A width of one generated image.

    height: int
        A height of one generated image.

    count: int
        A count of generated image.

    without_content: boolean
        A flag using to skip wrtiting a images content, just writing headers.
        It useful for testing DB classes in invalid data cases.
    """
    labels_db = MNISTLabelsFile()
    images_db = MNISTImagesFile(labels_db)

    labels_test_file = get_data_file_path(
        labels_db.file_name, data_home=target_dir)
    images_test_file = get_data_file_path(
        images_db.file_name, data_home=target_dir)

    with open(labels_test_file, 'wb') as labels:
        print("Write test labels data to '{}'".format(labels_test_file))
        with open(images_test_file, 'wb') as images:
            print("Write test images data to '{}'".format(images_test_file))
            # write a general header
            for f, db in [
                (labels, labels_db),
                (images, images_db),
            ]:
                f.write(struct.pack(db.header_format,
                                    db.header_magic_numer, count))
            # write ext headers
            images.write(struct.pack(
                images_db.image_header_format, width, height))
            # write content
            sz = width*height

            if not without_content:
                for i in range(count):
                    labels.write(struct.pack("B", i % 10))

                    data = np.zeros(sz, dtype=np.uint8)
                    data[0] = i % 10
                    images.write(data)

    print("Complete")


if __name__ == '__main__':
    """
    A tool generating test datafiles MNIST comparable.

    Example:
        python mnistdata.py ./test-data

    Requirements arguments:
        target dir
        A target directory to store testing datafiles.
    """
    if len(sys.argv) != 2:
        print("Expected a target dirname, but got more or less arguments\n")
        print("Using: python minstdata.py target_test_path\n")
        exit

    target_dir = sys.argv[1]

    width, height = 28, 28
    count = 20

    GenerateTestData(target_dir, width, height, count)
