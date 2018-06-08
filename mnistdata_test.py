import unittest
import os
import shutil

if __name__.find('.')<0:
    import mnistdata
else:
    from . import mnistdata

class TestMnistDownloader:
    """
    Mock of downloader is helping testing images and labels DB class.
    It is loading/copying files from one directory to another against downloading a file from URL.
    """

    base_path = ''

    def __init__(self, file_name):
        self.remote_file_name = os.path.join(
            TestMnistDownloader.base_path, file_name)
        if not os.path.exists(self.remote_file_name):
            raise Exception("File not found '" + self.remote_file_name + "'")

        self.file_size = os.path.getsize(self.remote_file_name)

        self.reader = open(self.remote_file_name, "rb")

    @property
    def remote_file_name(self):
        return self.__remote_file_name

    @remote_file_name.setter
    def remote_file_name(self, v):
        self.__remote_file_name = v

    @property
    def file_size(self):
        return self.__file_size

    @file_size.setter
    def file_size(self, v):
        self.__file_size = v

    def read(self):
        return self.reader.read(mnistdata.MNIST_READ_SIZE)

    def check_downloaded_size(self):
        pass

    def close(self):
        self.reader.close()


class TestMnistDataFetch(unittest.TestCase):
    test_download_path = "test-data/download-src"
    test_data_home_path = "test-data/data-home"

    def clear_dir(self):
        """
        Removing all directories from the list and all its files.
        """
        for dr in [TestMnistDataFetch.test_download_path, TestMnistDataFetch.test_data_home_path]:
            shutil.rmtree(dr, ignore_errors=True)

    def setUp(self):
        self.clear_dir()

        mnistdata.GenerateTestData(
            TestMnistDataFetch.test_download_path, 28, 28, 20)
        TestMnistDownloader.base_path = TestMnistDataFetch.test_download_path

        self.test_downloader = TestMnistDownloader
        self.labels_db = mnistdata.MNISTLabelsFile(
            downloader=self.test_downloader)
        self.images_db = mnistdata.MNISTImagesFile(
            self.labels_db, downloader=self.test_downloader)

    def tearDown(self):
        self.images_db.close()
        self.labels_db.close()

        self.clear_dir()

    def test_data_fetch(self):
        self.labels_db.fetch(
            data_home=TestMnistDataFetch.test_data_home_path)

        self.images_db.fetch(
            data_home=TestMnistDataFetch.test_data_home_path)

        for file_name in [self.labels_db.file_name, self.images_db.file_name]:
            self.assertTrue(os.path.exists(os.path.join(
                TestMnistDataFetch.test_data_home_path, file_name)))

    def test_data_read(self):
        mnistdata.GenerateTestData(
            TestMnistDataFetch.test_data_home_path, 28, 28, 20)

        self.labels_db.read(
            data_home=TestMnistDataFetch.test_data_home_path)
        self.images_db.read(
            data_home=TestMnistDataFetch.test_data_home_path)

        expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        for i in range(10):
            exist = sum(sum(self.images_db[i]))
            self.assertEqual(exist, expected[i])
    
    def test_data_read_fail(self):
        mnistdata.GenerateTestData(
            TestMnistDataFetch.test_data_home_path, 28, 28, 20, without_content=True)

        with self.assertRaises(Exception):
            self.labels_db.read(
                data_home=TestMnistDataFetch.test_data_home_path)

        with self.assertRaises(Exception):
            self.images_db.read(
                data_home=TestMnistDataFetch.test_data_home_path)

if __name__ == '__main__':
    unittest.main()
