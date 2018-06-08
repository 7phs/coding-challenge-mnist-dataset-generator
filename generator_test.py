import unittest
import shutil
import numpy as np

if __name__.find('.') < 0:
    import generator
    import filters
    import mnistdata
else:
    from . import generator
    from . import filters
    from . import mnistdata


class TestGenerator(unittest.TestCase):
    test_data_home_path = "test-data/data-home"

    def clear_dir(self):
        for dr in [TestGenerator.test_data_home_path]:
            shutil.rmtree(dr, ignore_errors=True)

    def setUp(self):
        self.clear_dir()

        mnistdata.GenerateTestData(
            TestGenerator.test_data_home_path, 28, 28, 20)

        self.labels_db = mnistdata.MNISTLabelsFile()
        self.images_db = mnistdata.MNISTImagesFile(self.labels_db)
        self.labels_db.read(
            data_home=TestGenerator.test_data_home_path)
        self.images_db.read(
            data_home=TestGenerator.test_data_home_path)

    def tearDown(self):
        self.images_db.close()
        self.labels_db.close()

        self.clear_dir()

    def test_generate_numbers_sequence(self):
        img = generator.generate_numbers_sequence([0, 2, 4, 6, 8], (3, 15), 160,
                                                  images=self.images_db)

        self.assertEqual(img.shape, (28, 160))


class TestParameter(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_default_parameters(self):
        test_cases = [
            ((28, 0, None, None), ((0, 0), 0)),
            ((28, 10, None, None), ((0, 0), 280)),
            ((28, 10, (3, 10), None), ((3, 10), 307)),
            ((28, 10, (3, 10), 230), ((3, 10), 230)),
        ]

        for test, expected in test_cases:
            self.assertEqual(generator.default_parameters(*test), expected)

    def test_check_aparameters(self):
        test_cases = [
            ((0, (0, 0), 250), True),
            ((5, (5, 0), 250), True),
            ((5, (-5, 0), 250), True),
            ((5, (-10, -4), 250), True),
            ((10, (10, 20), 80), True),
            ((10, (3, 10), -310), True),
            ((10, (3, 10), 0), True),
            ((10, (3, 10), 310), False),
        ]

        for test, expected in test_cases:
            if expected:
                with self.assertRaises(Exception):
                    generator.check_parameters(*test)
            else:
                generator.check_parameters(*test)

    def test_get_filters(self):
        test_cases = [
            ((28, 255, 10, (3, 10), 230, True, None),
                ([
                    "numpy.lib.function_base.vectorize",
                    "numpy.lib.function_base.vectorize",
                    "function resize_seq.<locals>.resize_image",
                    "function spacing_seq.<locals>.add_spacing",
                ],
                [])
            ),
            ((28, 255, 10, (3, 10), 230, True, [
                filters.blur(),
                filters.distort(20),
            ]), ([
                "numpy.lib.function_base.vectorize",
                "numpy.lib.function_base.vectorize",
                "function blur.<locals>.blur_image",
                "function distort.<locals>.distort_image",
                "function resize_seq.<locals>.resize_image",
                "function spacing_seq.<locals>.add_spacing",
                ],
                [])
            ),
            ((28, 255, 10, (3, 10), 230, False, None), 
                ([
                "numpy.lib.function_base.vectorize",
                "numpy.lib.function_base.vectorize",
                "function resize_seq.<locals>.resize_image",
                "function spacing_seq.<locals>.add_spacing",
                ],
                [
                "function resize.<locals>.resize_image",
                ])
            ),
            ((28, 255, 10, (3, 10), 230, False, [
                filters.blur(),
                filters.distort(20),
            ]), ([
                "numpy.lib.function_base.vectorize",
                "numpy.lib.function_base.vectorize",
                "function resize_seq.<locals>.resize_image",
                "function spacing_seq.<locals>.add_spacing",
                ],
                [
                "function blur.<locals>.blur_image",
                "function distort.<locals>.distort_image",
                "function resize.<locals>.resize_image",
                ])
            ),
        ]

        for test, expected in test_cases:
            exist_processing, exist_postprocessing = generator.get_filters(
                *test)

            expected_processing, expected_postprocessing = expected

            self.assertEqual(len(exist_processing), len(expected_processing))

            for t, e in zip(exist_processing, expected_processing):
                self.assertTrue(repr(t).find(
                    e) >= 0, msg="unexpected filter {}, expected is {}".format(repr(t), e))

            self.assertEqual(len(exist_postprocessing), len(expected_postprocessing))

            for t, e in zip(exist_postprocessing, expected_postprocessing):
                self.assertTrue(repr(t).find(
                    e) >= 0, msg="unexpected filter {}, expected is {}".format(repr(t), e))


if __name__ == '__main__':
    unittest.main()
