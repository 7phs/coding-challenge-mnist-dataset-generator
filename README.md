# MNIST dataset generator

An MNIST dataset generator is a tool for creating images using to train classifiers and generative deep learning models.

A tool is using data files of [MNIST database](http://yann.lecun.com/exdb/mnist/).
It is prepared set of images of handwritten digits.
That files will automatically download at the first time and store in the local caching directory.
Downloading will repeating if the data files remove from the directory.

The generator is implemeting both way tools: a console utility and a library which can use in your projects.

All of image processing method of the library is implemented as a **filter function** - resizing, add spaces, etc.
User of the library can extend processing using own implemented **filter function**.

A result of the console utilities is a file stored in PNG format.
A result of the library is a NumPy array.

The library is implementing additional methods of data expansion - blurring and distorting.
Developers can implement own method and use it for image transformation.

## Requirements

An MNIST dataset generator was developing and testing on **Python 3.6+**.
It is using a several 3rd party libraries.


### Dependencies

1. [numpy](http://www.numpy.org) ver. 1.14

    **NumPy** is the fundamental package for scientific computing with Python.
    It provides wide area method to process data.

    Install: ```pip install numpy```

2. [scipy](https://scipy.org) 1.0

    The **SciPy** library, a collection of numerical algorithms and domain-specific toolboxes, including signal processing, optimization, statistics and much more.

    It is using for additional image manipulation such as bluring and will using in implementing another methods.

    Install: ```pip install scipy```

3. [scikit-image](http://scikit-image.org) 0.13.1

    **scikit-image** is a collection of algorithms for image processing.

    It is using to resize an image while processing it.

    **scipy.misc.imresize** is deprecated in SciPy 1.0.0, and will be removed in 1.2.0.
    **SciPy** doc recomends using **skimage.transform.resize** instead.

    Install: ```pip install scikit-image```

4. [imageio](https://imageio.github.io) 2.3

    Imageio is a Python library that provides an easy interface to read and write a wide range of image data.

    It is using to store an array as a PNG image.

    **scipy.misc.imsave**  is deprecated in SciPy 1.0.0, and will be removed in 1.2.0.
    **SciPy** doc recomends using **imageio.imwrite** instead.

    Install: ```pip install imageio```

5. [tqdm](https://tqdm.github.io) 4.23

    A fast, extensible progress bar for Python and CLI.
    It makes as easy as possible to show console progress just adding one line.
    It was using to show progress of downloading MNIST datafiles.

    **tqdm** was widly using in the [Udacity](https://www.udacity.com) deep learning educational projects

    Install: ```pip install tqdm```



## Tools: "Dataset Generator"

A tool generating a PNG image from a digit sequence using random prepared handwritten symbols of MNIST database.

**Usages**:

```bash
python generator.py [-h]
                    [-o image_name.png]
                    [-d DATA_DIRECTORY]
                    [-w IMAGE_WIDTH]
                    [-s min,max]
                    [-e]
                    [-f filter1,filter2]
                    digits
```

**Example of running**:

Shortest:

```bash
python generator.py 498127864687234
```

With parameters:

```bash
python generator.py -o first_image.png -w 360 -s 0,20 -f distort 498127864687234
```

### Required arguments:

**digits**

A string each digit characters which sequenced transformed to an image using MNIST images.

### Optiononal arguments:

**-o | --output**
    
Default: mnist_numbers_sequence.png

A name of a result PNG image.

**-d | --data_directory**

A custom path to cache MNIST datafile.
A default data directory **~/generator_numbers_seq_mnist** will use if the parameter skipped.

**-w | --image_width**

A width of a result image.

**-s | --spacing**

Default: ```0,0```

A range of spacing between digit images separated a comma.

Format: ```minimum,maximum```

Example:

- ```3```

- ```2,100```

**-e | --evenly**

An evenly spaced of spacing against a default randomly choosen in the spacing range.

**-f | --filters**

A list of additional filters applied each digit image.
Filter applied by order of value.

Supported filters:

- ```blur``` a little bit blurring an image
- ```distort``` make a random horizontal moving each line of an image.

Format: ```filter1,filter2```

Example:

- ```blur```
- ```distort,blur```
- ```blur,distort```

### Chaching data

The MNIST datafiles will download if not finding in local chaching directory.

A default chaching directory is **~/generator_numbers_seq_mnist**.



## Tools: "Test Dataset Generator"

Another tool is a generator of test datafiles MNIST comparable.
It useful to test the library code.

**Usages**:

```bash
python mnistdata.py target_directory
```

**Example of running**:

```bash
python mnistdata.py ./test-data
```

### Required arguments:

**target_directory**

A target directory to store testing datafiles.



## API

A library might be using in the 3rd party project.
Example, as one of the early stage of data workflow to generating traning and testing data sets.

### Library Configuration

A MNIST datafile will automatically downloading at the first call of generator API.
Default caching directory is **~/generator_numbers_seq_mnist**. 
You should set an environment parameter **GENERATOR_NUMBERS_SEQ_MNIST_DIR** to change that location.

### Generator

Generator API method has the same parameters as a tool.

```python
def generate_numbers_sequence(digits,
                              spacing_range,
                              image_width,
                              data_home=None,
                              images=None,
                              evenly=False,
                              fltrs=None):
```

**Example of usage**

```python
from mnist_dataset_generator import generator, filters

img = generator.generate_numbers_sequence(
    [4, 5, 6, 2, 6, 1, 3, 2], (0, 10), 350,
    fltrs=[filters.blur(), filters.distort(10)])
```

#### Parameters

**digits**

A list-like containing the numerical values of the digits from which the sequence
will be generated (for example [3, 5, 0]). A number greater 10 will reducing to be less than 10.

**spacing_range**

A (minimum, maximum) pair (tuple), representing the min and max spacing between digits.
A unit should be a pixel.

**image_width**

Specifies the width of the image in pixels.

**data_home** Optional

A custom path of storing MNIST datafiles.

**images** Optional

A custom MNIST image db to prevent using default DB of a mnistdata module.
Ex. it useful for testing.

**evenly** Optional

A mode of generating an image.
If False - Randomly choosing a spacing in the spacing_range.
If True - evenly interval for each image and spacing.

**fltrs** Optional

A list-like containing functions. Each of them will apply on a digit image and modify it
before adding to sequence.

#### Returns

The image containing the sequence of numbers. The image is representing
as floating point 32bits numpy arrays with a scale ranging from 0 (black) to 1 (white).

### Filters


A user of the library can extend a method of processing an image implementing own **filter functions**.

A **filter function** has one parameter - NumPy array with a shape (28, *) and the same type return.

One of the filter requirements is not changing a Y-axes.

Add an implemented **filter funcion** to a list of the **fltr** parameters of calling **generate_numbers_sequence**.

**Example of a filter function**:

```python
from scipy.ndimage.filters import gaussian_filter

def blur_image(img):
    return gaussian_filter(img, sigma=13)
```


## Trade-off

The implemented generator has a several trade-off:

1. **Validation** of downloaded datafiles.

    Files received from remote resources should include validation.
    The current validation is just checking a magic number of a file header,
    another method is comparing file size and expected size based records count from a file header.
    The current validation is based on the getting from the resources information and
    doesn't using additional library information.

    A good way to validation is comparing **checksum** of the downloaded file and received from the source websites.
    Updated of **checksum** requires implementing an especial code infrastructure.
    It wasn't implemeted in the current solution. This decision depens on time restriction.

2. **File** is a primary source of the data.

    The current solution is reading image data for processing each time.
    It is useful for the restricted environment and in the case of the unknown size of a data file.

    SDD or a hard drive is a bottleneck of the processing data traffic
    The best way is loaded all source data in memory if it is possible.
    Another way is using a different caching strategy for loading in memory.
    It is more complicated and needing to gathering additional requirements.

    Another reason for using memory the next trade-off point - parallelization of processing image.
    File storage is the very tight bottleneck in this case.

3. **Only single** thread/process using in a workflow.

    Image processing is highly loading operations.
    Using all of the CPU cores or GPU is the good fit to grow up a performance of the solution.

    Using Python modules such as **multiprocessing** or **threading** is a good chance solve this trade-off point.
    It requires needing to gathering additional requirements.

4. **Simpliest** logging method.

    Python print function is the only using method for logging events.
    It is restricted approach without a possibility to configure it.

    The best way is using dependencies injection.
    For example, adding a specific parameter for a generate method.

    Another point is the progress bar in downloading.
    It is just using only for console and needing to make configurable.