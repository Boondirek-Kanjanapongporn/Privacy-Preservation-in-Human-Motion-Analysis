from os import listdir
from os.path import isfile, join
import numpy as np
import re
from datapreprocessing import *

# For datapreprocessing.py ------------------------------------------
def getfilepath1(filename):
    return 'src/raw_data/' + filename

def getfilepath2(folder, filename):
    return 'D:/Individual Project/' + folder + '/' + filename

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
    
# For main.py ------------------------------------------
def getfilenamesfromdir(folder):
    path = f"D:/Individual Project/{folder}"
    return [f for f in listdir(path) if isfile(join(path, f))]

def check_consistent_activity(filename):
    k, xx, yy, z = re.split('P|A|R', filename)
    if int(k) == int(yy):
        return True
    else:
        # print(f"k: {k}\nxx: {xx}\nyy: {yy}\nz: {z}")
        print(filename)
        return False

def are_folders_clear(folders):
    for folder in folders:
        print(folder + ":")
        isclear = True
        filenames = getfilenamesfromdir(f"D:/Individual Project/{folder}")
        for filename in filenames:
            if not check_consistent_activity(filename):
                isclear = False
        print("isClear: " + str(isclear))

def store_dataset_by_dir(folders):
    for folder in folders:
        activities_dataset = []
        print(folder + ":")
        filenames = getfilenamesfromdir(folder)
        for filename in filenames:
            data = preprocess(folder, filename, False, False, True)
            if data is not None:
                activities_dataset.append(data)
                print(filename)
            else:
                print(f"{filename}: Failed")
        np.save(f"{folder}.npy", activities_dataset)
        print("Done Saving!!")

def store_label_by_dir(folders):
    for folder in folders:
        activities_label = []
        print(folder + ":")
        filenames = getfilenamesfromdir(folder)
        for filename in filenames:
            # Only for "6 February 2019 NG Homes Dataset"
            # if filename in ("1P20A01R1", "1P20A01R2", "1P20A01R3", "1P21A01R1", "1P21A01R2", "1P26A01R1", "1P26A01R2", "1P26A01R3"):
            #     continue
            activities_label.append(int(filename[0]))
        np.save(f"{folder[:-7]}Label.npy", activities_label)
        print("Done Saving!!")