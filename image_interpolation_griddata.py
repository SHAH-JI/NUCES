% image interpolation using scipy.interpolation.griddata
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import pandas as pd
from scipy.misc import imsave
def make_interpolated_image(data):
    ix = data["x"]
    iy = data["y"]
    samples =data["f(x,y)"]
    int_im = griddata((ix, iy), samples, (X, Y),fill_value=0)
    return int_im

data = pd.read_csv( "data.csv")
#print (data.describe())
data=data.sort_values(by=["x","y"],axis=0)
data =data.reset_index(drop=True)
#print (np.where(data["x"]==100)[0][0]-1)


X, Y = np.meshgrid(np.arange(0, data["x"].max()+1, 1), np.arange(0, data["y"].max()+1, 1))
matrix =make_interpolated_image(data)
for i in range(1000):
    for j in range(2000):
        matrix[i][j]= matrix[i][j] % 256
imsave('my.png',matrix)
