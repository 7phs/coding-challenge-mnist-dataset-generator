import unittest
import numpy as np

if __name__.find('.') < 0:
    import filters
else:
    from . import filters


class TestMnistDataFetch(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_invert(self):
        inverter = filters.invert(9)

        expected = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
        exist = inverter(list(range(10)))

        self.assertTrue(np.array_equal(exist, expected))

    def test_normalize(self):
        mormalizer = filters.normalize(10)

        expected = np.array([.0, .1, .2, .3, .4, .5, .6, .7, .8, .9])
        exist = mormalizer([i for i in range(10)])

        self.assertTrue(np.array_equal(exist, expected))

    def test_resize_seq(self):
        resizer = filters.resize_seq((1+i*2 for i in range(10)), 0)

        img = np.zeros(shape=(28, 1), dtype=np.uint8)

        for _ in range(10):
            img = resizer(img)

        self.assertEqual((28, 19), img.shape)

    def test_spacing_seq(self):
        spacer = filters.spacing_seq((i for i in range(10)), 0)

        img = np.zeros(shape=(28, 0), dtype=np.uint8)

        for _ in range(10):
            img = spacer(img)

        self.assertEqual((28, 45), img.shape)

    def test_resize(self):
        resizer = filters.resize(250)

        img = np.zeros(shape=(28, 1), dtype=np.float32)

        img = resizer(img)

        self.assertEqual((28, 250), img.shape)


if __name__ == '__main__':
    unittest.main()
