from os import listdir
from os.path import isfile, join
import numpy as np
import re

RAWDATAFOLDER = "D:/Individual Project/Data"
PREPROCESSEDFOLDER = "D:/Individual Project/Preprocessed Data2"

def getfilenamesfromdir(folder):
    path = f"{RAWDATAFOLDER}/{folder}"
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
        filenames = getfilenamesfromdir(f"{RAWDATAFOLDER}/{folder}")
        for filename in filenames:
            if not check_consistent_activity(filename):
                isclear = False
        print("isClear: " + str(isclear))

# For Preprocessed Data ------------------------------------------------------
# Store dataset to folder
# folders = ['1 December 2017 Dataset', '2 March 2017 Dataset', '3 June 2017 Dataset', '4 July 2018 Dataset', '5 February 2019 UoG Dataset', '6 February 2019 NG Homes Dataset', '7 March 2019 West Cumbria Dataset']
# for folder in folders:
#     print(folder + ":")
#     filenames = getfilenamesfromdir(folder)
#     for filename in filenames:
#         data = preprocess(folder, filename, False, False, True, True)
#         np.save(f"{PREPROCESSEDFOLDER}/{folder} Normalized/{filename[:-4]} Normalized.npy", data)
#         print(filename)
#     print("Done Saving!!")

def vstack_preprocessdata(folder, name):
    datalist = []
    filenames = getfilenamesfromdir(folder)
    for filename in filenames:
        # Repetition == 1
        # if int(re.split("R| ", filename)[1].split('.')[0]) == 1:
        print(filename + ": ")
        datalist.append(np.load(f"{PREPROCESSEDFOLDER}/{folder} Normalized/{filename[:-4]} Normalized.npy"))
    print("Storing...")
    print(np.array(datalist).shape)
    np.save(f"{PREPROCESSEDFOLDER}/{name}.npy", np.array(datalist))
    print("Done Saving!!")

def vstack_datasets(folders, name):
    vstack_datasets = None
    for folder in folders:
        print(folder + ": ")
        if vstack_datasets is None:
            vstack_datasets = np.load(f"{PREPROCESSEDFOLDER}/{folder}.npy")
        else:
            dataset = np.load(f"{PREPROCESSEDFOLDER}/{folder}.npy")
            vstack_datasets = np.vstack((vstack_datasets, dataset))
    print("Storing...")
    print(vstack_datasets.shape)
    np.save(f"{PREPROCESSEDFOLDER}/{name}.npy", vstack_datasets)
    print("Done Saving!!")

def store_labels(folders):
    for folder in folders:
        activities_label = []
        print(folder + ":")
        filenames = getfilenamesfromdir(folder)
        for filename in filenames:
            # Repetition == 1
            # if int(re.split("R| ", filename)[1].split('.')[0]) == 1:
            activities_label.append(int(filename[0]) - 1)
        # np.save(f"{PREPROCESSEDFOLDER}/{folder[:-8]} Label R1.npy", np.array(activities_label))
        np.save(f"{PREPROCESSEDFOLDER}/{folder[:-8]} Label.npy", np.array(activities_label))
        print("Done Saving!!")

def vstack_labels(folders, name):
    vstacklabel = None
    for folder in folders:
        print(folder + ": ")
        if vstacklabel is None:
            # vstacklabel = np.load(f"{PREPROCESSEDFOLDER}/{folder[:-8]} Label R1.npy")
            vstacklabel = np.load(f"{PREPROCESSEDFOLDER}/{folder[:-8]} Label.npy")
        else:
            # label = np.load(f"{PREPROCESSEDFOLDER}/{folder[:-8]} Label R1.npy")
            label = np.load(f"{PREPROCESSEDFOLDER}/{folder[:-8]} Label.npy")
            vstacklabel = np.concatenate((vstacklabel, label))
    print("Storing...")
    print(vstacklabel.shape)
    np.save(f"{PREPROCESSEDFOLDER}/{name}.npy", vstacklabel)
    print("Done Saving!!")