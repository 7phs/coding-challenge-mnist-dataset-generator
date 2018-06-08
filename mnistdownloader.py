import io
from tqdm import tqdm
import zlib
from urllib import error, request, parse

if __name__.find('.')<0:
    import helper
else:
    from . import helper

MNIST_DATASET_URL = "http://yann.lecun.com/exdb/mnist/"
MNIST_DATASET_URL_EXT = ".gz"

MNIST_DOWNLOAD_SIZE = 64 * 1024

class HttpDownloader:
    def __init__(self, file_name):
        """
        A helper of downloading an MNIST datafile.
        A parameters (URL, etc.) stores on the module level in constants
        A progress bar will show a process of downloading.

        Parameters
        ----------
        file_name : str
            A local name of a file which downloading.
        """
        # prepare a request parameters
        self.remote_file_name = file_name + MNIST_DATASET_URL_EXT

        url = parse.urljoin(MNIST_DATASET_URL, self.remote_file_name)
        req = request.Request(url)
        # request a data file
        try:
            self.resp = request.urlopen(req)
        except error.HTTPError as e:
            print("HTTPError: ", e)
            raise e

        headers = dict(self.resp.info())
        self.expected_size = int(headers["Content-Length"])
        self.downloaded_size = 0
        self.progress_bar = None

        self.file_size = helper.file_size(self.expected_size)

        self.decompresser = zlib.decompressobj(zlib.MAX_WBITS | 32)
        self.reader = io.BufferedReader(self.resp)
    
    @property
    def remote_file_name(self):
        """
        A name of a downloadable file property
        """
        return self.__remote_file_name
    
    @remote_file_name.setter
    def remote_file_name(self, v):
        """
        A setter of a name of a downloadable file.

        Parameters
        ----------
        v: str
            The new name of a file.
        """
        self.__remote_file_name = v
    
    @property
    def file_size(self):
        """
        A downloaded file size property.
        """
        return self.__file_size
    
    @file_size.setter
    def file_size(self, v):
        """
        A setter of downloaded file size property.

        Parameters
        ----------
        v: int
            The new size of downloadable file.
        """
        self.__file_size = v

    def read(self):
        """
        Reding and uncompress the next piece of datafiles.

        Returns
        -------
        An array of uncopressed data.
        """
        if not self.progress_bar:
            self.progress_bar = tqdm(total=self.expected_size, unit="bytes", unit_scale=True)

        data = self.reader.read(MNIST_DOWNLOAD_SIZE)
        if not data:
            return data

        self.downloaded_size += len(data)
        self.progress_bar.update(len(data))

        return self.decompresser.decompress(data)

    def check_downloaded_size(self):
        """
        Checking exists and expected count of downloaded bytes.
        It helps prevent an unexpected error while downloading.

        Raises
        ------
        An exception containing a string description of an error.
        """
        if self.expected_size!=self.downloaded_size:
            raise Exception("file wasn't downloading success. Received {}, but expected {}".format(self.downloaded_size, self.expected_size))

    def close(self):
        """
        Closing a progress bar of a downloading process if an object was using it.
        """
        if self.progress_bar:
            self.progress_bar.close()
