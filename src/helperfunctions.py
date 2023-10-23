from os import listdir
from os.path import isfile, join
import numpy as np
import re
from datapreprocessing import *

# For datapreprocessing.py ------------------------------------------
def getfilepath1(filename):
    return 'src/raw_data/' + filename

def getfilepath2(folder, filename):
    return 'D:/Individual Project/Data/' + folder + '/' + filename

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
rawdatapath = "D:/Individual Project/Data"
preprocessdatapath = "D:/Individual Project/Preprocessed Data"
preprocessdatapath2 = "D:/Individual Project/Preprocessed Data2"

def getfilenamesfromdir(folder):
    path = f"{rawdatapath}/{folder}"
    return [f for f in listdir(path) if isfile(join(path, f))]

def check_consistent_activity(filename):
    k, _, yy, _ = re.split('P|A|R', filename)
    if int(k) == int(yy):
        return True
    else:
        print(filename)
        return False

def are_folders_clear(folders):
    for folder in folders:
        print(folder + ":")
        isclear = True
        filenames = getfilenamesfromdir(f"{rawdatapath}/{folder}")
        for filename in filenames:
            if not check_consistent_activity(filename):
                isclear = False
        print("isClear: " + str(isclear))

# For Preprocessed Data ------------------------------------------------------
def store_preprocessdata1(folders):
    for folder in folders:
        activities_dataset = []
        print(folder + ":")
        filenames = getfilenamesfromdir(folder)
        for filename in filenames:
            activities_dataset.append(preprocess(folder, filename, False, False, True, False))
            print(filename)
        np.save(f"{preprocessdatapath}/{folder}.npy", np.array(activities_dataset))
        print("Done Saving!!")

def store_labels1(folders):
    for folder in folders:
        activities_label = []
        print(folder + ":")
        filenames = getfilenamesfromdir(folder)
        for filename in filenames:
            activities_label.append(int(filename[0]))
        np.save(f"{preprocessdatapath}/{folder[:-7]}Label.npy", np.array(activities_label))
        print("Done Saving!!")

def vstack_preprocessdata1(folders, name):
    vstackdataset = None
    for folder in folders:
        print(folder + ": ")
        if vstackdataset is None:
            vstackdataset = np.load(f"{preprocessdatapath}/{folder}.npy")
        else:
            dataset = np.load(f"{preprocessdatapath}/{folder}.npy")
            vstackdataset = np.vstack((vstackdataset, dataset))
    print("Storing...")
    np.save(f"{preprocessdatapath}/{name}.npy", vstackdataset)
    print("Done Saving!!")

def vstack_labels1(folders, name):
    vstacklabel = None
    for folder in folders:
        print(folder + ": ")
        if vstacklabel is None:
            vstacklabel = np.load(f"{preprocessdatapath}/{folder[:-7]}Label.npy")
        else:
            label = np.load(f"{preprocessdatapath}/{folder[:-7]}Label.npy")
            vstacklabel = np.concatenate((vstacklabel, label))
    print("Storing...")
    np.save(f"{preprocessdatapath}/{name}.npy", vstacklabel)
    print("Done Saving!!")

def normalize_dataset(file):
    dataset = np.load(f"{preprocessdatapath}/{file}.npy")
    normalized_list = []
    for data_array in dataset:
        data_min = np.min(data_array)
        data_max = np.max(data_array)

        normalized_data = (data_array - data_min) / (data_max - data_min)
        normalized_list.append(normalized_data)
    print("Storing...")
    np.save(f"{preprocessdatapath}/{file} Normalized.npy", np.array(normalized_list))
    print("Done Saving!!")

# For Preprocessed Data 2 ------------------------------------------------------
def store_preprocessdata2(folders):
    for folder in folders:
        print(folder + ":")
        filenames = getfilenamesfromdir(folder)
        for filename in filenames:
            np.save(f"{preprocessdatapath2}/{folder} Normalized/{filename[:-4]} Normalized.npy", preprocess(folder, filename, False, False, True, True))
            print(filename)
        print("Done Saving!!")

def vstack_preprocessdata2(folder, name):
    datasetlist = []
    filenames = getfilenamesfromdir(folder)
    for filename in filenames:
        print(filename + ": ")
        datasetlist.append(np.load(f"{preprocessdatapath2}/{folder} Normalized/{filename[:-4]} Normalized.npy"))
    print("Storing...")
    np.save(f"{preprocessdatapath2}/{name}.npy", np.array(datasetlist))
    print("Done Saving!!")

def vstack_datasets2(folders, name):
    vstackdataset = None
    for folder in folders:
        print(folder + ": ")
        if vstackdataset is None:
            vstackdataset = np.load(f"{preprocessdatapath2}/{folder}.npy")
        else:
            dataset = np.load(f"{preprocessdatapath2}/{folder}.npy")
            vstackdataset = np.vstack((vstackdataset, dataset))
    print("Storing...")
    print(vstackdataset.shape)
    np.save(f"{preprocessdatapath2}/{name}.npy", vstackdataset)
    print("Done Saving!!")

def store_labels2(folders):
    for folder in folders:
        activities_label = []
        print(folder + ":")
        filenames = getfilenamesfromdir(folder)
        for filename in filenames:
            activities_label.append(int(filename[0]))
        np.save(f"{preprocessdatapath2}/{folder[:-8]} Label.npy", np.array(activities_label))
        print("Done Saving!!")

def vstack_labels2(folders, name):
    vstacklabel = None
    for folder in folders:
        print(folder + ": ")
        if vstacklabel is None:
            vstacklabel = np.load(f"{preprocessdatapath2}/{folder[:-8]} Label.npy")
        else:
            label = np.load(f"{preprocessdatapath2}/{folder[:-8]} Label.npy")
            vstacklabel = np.concatenate((vstacklabel, label))
    print("Storing...")
    print(vstacklabel.shape)
    np.save(f"{preprocessdatapath2}/{name}.npy", vstacklabel)
    print("Done Saving!!")