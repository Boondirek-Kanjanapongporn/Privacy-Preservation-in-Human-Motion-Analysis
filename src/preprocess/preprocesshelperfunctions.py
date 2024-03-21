import numpy as np
from pathlib import Path

PREPROCESSEDFOLDER = "../../data/raw"

def getfilepath1(filename):
    return 'data/raw/1 December 2017 Dataset Normalized' + filename

def getfilepath2(folder, filename):
    # return 'D:/Individual Project/Data/' + folder + '/' + filename
    script_location = Path(__file__).resolve().parent
    data_folder = script_location / PREPROCESSEDFOLDER
    data_path = data_folder / folder / filename
    return data_path

def complex_converter(s):
    if 'i' in s:
        return complex(s.replace('i', 'j'))
    else:
        return float(s)

def oddnumber(x):
    if np.isscalar(x):  # Check if x is a scalar
        y = np.floor(x)
        if y % 2 == 0:
            y = np.ceil(x)
        if y % 2 == 0:
            y = y + 1
        return int(y)  # Return scalar as int
    
    else:  # If x is an array
        y = np.zeros_like(x)
        for k in range(len(x)):
            y[k] = np.floor(x[k])
            if y[k] % 2 == 0:
                y[k] = np.ceil(x[k])
            if y[k] % 2 == 0:
                y[k] = y[k] + 1
        return y  # Return array