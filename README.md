This project is developed for Python3.5 interpreter on linux machine. Using Anaconda virtual environment is recommended.

To install dependencies, simply run:

```pip install -r requirment.txt```

This project uses TensorFlow, a machine learning library developed and maintained by Google in principle.

We use tensorflow version 1.4.0 ***(required)***,

```pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.0-cp35-cp35m-linux_x86_64.whl```

users can choose to install its GPU optimized version accordingly,

```pip install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0-cp35-cp35m-linux_x86_64.whl```

To install cv2 in Anaconda (optional):

```conda install -c menpo opencv=2.4.11```

Or via pip:

```pip install opencv-python```

## Datat set

Small galaxy zoo is available at [https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data). I downloaded and used only the training images and training labels.

To see how I select images that has either elliptical or sprial galaxies, checkout the jupyter notebook in ```data``` folder.

To see how does the overlapping work, chekcout the ```python galaxy_data.py``` once you have the data set ready.

## Train & Test

Simply,

```python galaxy_main.py```
