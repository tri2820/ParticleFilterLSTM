import numpy as np
import os
import matplotlib.pyplot as plt

def plot_sample(measure_series):
    """measure_series is a series of a single measurement, 
    at each time step, contains 
    an axis Tensor(x,y,z,..) 
    or an sensor Tensor(value_sensor0, value_sensor1, ...)
    """
    x_series = measure_series[:,0]
    plt.plot(x_series)
    


def list_all_data():
    for root, dirs, files in os.walk("data"):
        for file in sorted(files):
            filepath = os.path.join(root, file)
            data = np.load(filepath)

            print(filepath)
            print('\t',data.shape)


def fig(filepaths):
    if type(filepaths) == str:
        filepaths = [filepaths]
    for filepath in filepaths:
        data = np.load(filepath)
        plot_sample(data[0])
    return plt.show


# Example shape represents 
# 4096 runs of a 1D target, with 1 sensor, measured in 50 time steps
# (4096,50,1,1)

# There is no explaination for the third dimension, maybe for multitarget later
# Consider it as dummy dimension for now

fig([
    "data/train-data-measurements.npy",
    "data/train-data-ground_truth.npy"
    ])()




