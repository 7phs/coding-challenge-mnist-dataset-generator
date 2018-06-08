import unittest
import os
import shutil

if __name__.find('.')<0:
    import helper
else:
    from . import helper


class TestFileSize(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_file_size(self):
        test_cases = [
            (768, "768 bytes"),
            (10234, "9.994 KiB"),
            (98710234, "94.14 MiB"),
        ]

        for test, expected in test_cases:
            exist = helper.file_size(test)
            self.assertEqual(exist, expected)


class TestInterval(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_evenly_interval(self):
        test_cases = [
            ((10, 3), [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]),
            ((5, 0), [0, 0, 0, 0, 0]),
            ((0, 5), []),
        ]

        for test, expected in test_cases:
            exist = list(helper.evenly_interval(*test))
            self.assertEqual(exist, expected)

    def test_most_evenly_interval(self):
        test_cases = [
            ((9, 3, 35), [4, 4, 4, 4, 4, 4, 4, 3, 4]),
            ((9, 3, 30), [4, 3, 4, 3, 4, 3, 3, 3, 3]),
            ((9, 3, 28), [4, 3, 3, 3, 3, 3, 3, 3, 3])
        ]

        for test, expected in test_cases:
            exist = list(helper.most_evenly_interval(*test))
            self.assertEqual(exist, expected)

    def test_random_interval(self):
        test_cases = [
            (9, (3,  35)),
            (9, (2,  3)),
            (9, (10, 10)),
            (9, (0,  0)),
        ]

        for test in test_cases:
            exist = list(helper.random_interval(*test))
            minimum, maximum = test[1]
            
            self.assertTrue(all([v>=minimum and v<=maximum for v in exist]))

class TestIntervalCalc(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_randomly_case(self):
        digit_width = 28
        digit_count = 10
        unused_v = 200
        spacing = (10, 20)
        
        width_seq, spacing_seq = helper.randomly_image_interval(digit_width, digit_count, unused_v, spacing)

        exist_width = list(width_seq)
        exist_spacing = list(spacing_seq)

        minimum, maximum = spacing

        self.assertEqual(sum(exist_width), digit_count * digit_width)
        self.assertTrue(all([v>=minimum and v<=maximum for v in exist_spacing]))

    def test_evenly_case(self):
        test_cases = [
            (
                {'digit_width': 28, 'digit_count': 10,
                    'image_width': 307, 'spacing': (3, 10)},
                (
                    [28, 28, 28, 28, 28, 28, 28, 28, 28, 28],
                    [3, 3, 3, 3, 3, 3, 3, 3, 3]
                )
            ),
            (
                {'digit_width': 28, 'digit_count': 10,
                    'image_width': 310, 'spacing': (3, 10)},
                (
                    [28, 28, 28, 28, 28, 28, 28, 28, 28, 28],
                    [4, 3, 4, 3, 4, 3, 3, 3, 3]
                )
            ),
            (
                {'digit_width': 28, 'digit_count': 10,
                    'image_width': 316, 'spacing': (3, 10)},
                (
                    [28, 28, 28, 28, 28, 28, 28, 28, 28, 28],
                    [4, 4, 4, 4, 4, 4, 4, 4, 4]
                )
            ),
            (
                {'digit_width': 28, 'digit_count': 10,
                    'image_width': 279, 'spacing': (3, 10)},
                (
                    [25, 25, 25, 25, 25, 25, 25, 25, 25, 25],
                    [4, 3, 4, 3, 3, 3, 3, 3, 3]
                )
            ),
            (
                {'digit_width': 28, 'digit_count': 10,
                    'image_width': 160, 'spacing': (3, 10)},
                (
                    [13, 13, 13, 13, 13, 13, 13, 13, 13, 13],
                    [4, 3, 4, 3, 4, 3, 3, 3, 3]
                )
            ),
            (
                {'digit_width': 28, 'digit_count': 10,
                    'image_width': 563, 'spacing': (3, 10)},
                (
                    [48, 48, 48, 48, 48, 48, 48, 48, 48, 48],
                    [10, 9, 10, 9, 9, 9, 9, 9, 9]
                )
            ),
            (
                {'digit_width': 28, 'digit_count': 10,
                    'image_width': 563, 'spacing': (3, 3)},
                (
                    [54, 54, 53, 54, 53, 54, 53, 54, 53, 54],
                    [3, 3, 3, 3, 3, 3, 3, 3, 3]
                )
            ),
        ]

        for test, expected in test_cases:
            width_seq, spacing_seq = helper.evenly_image_interval(**test)
            exist_spacing = list(spacing_seq)
            exist_width = list(width_seq)

            expected_width, expected_spacing = expected

            self.assertEqual(exist_width, expected_width)
            self.assertEqual(exist_spacing, expected_spacing)
            self.assertEqual(sum(exist_width) +
                             sum(exist_spacing), test['image_width'])


class TestNotExistsFileName(unittest.TestCase):
    test_dir = "test-data/exists-file-name"

    def clearDir(self):
        shutil.rmtree(TestNotExistsFileName.test_dir, ignore_errors=True)

    def makeDir(self):
        os.mkdir(TestNotExistsFileName.test_dir)

    def setUp(self):
        self.clearDir()
        self.makeDir()

    def tearDown(self):
        self.clearDir()

    def test_not_exists_file_name(self):
        expected = os.path.join(TestNotExistsFileName.test_dir, "test.file")
        
        exist = helper.not_exists_file_name(expected)

        self.assertEqual(exist, expected)

        with open(expected, "w") as _:
            pass
        
        exist = helper.not_exists_file_name(expected)

        self.assertNotEqual(exist, expected)

if __name__ == '__main__':
    unittest.main()
