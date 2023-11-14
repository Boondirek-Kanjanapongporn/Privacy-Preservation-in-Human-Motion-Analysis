from os import listdir
from os.path import isfile, join
import numpy as np
import re

RAWDATAFOLDER = "D:/Individual Project/Data"
PREPROCESSEDFOLDER = "D:/Individual Project/Preprocessed Activity Data"
PREPROCESSEDFOLDER2 = "D:/Individual Project/Preprocessed Participant Data"

def getfilenamesfromdir(path, folder):
    fullpath = f"{path}/{folder}"
    return [f for f in listdir(fullpath) if isfile(join(fullpath, f))]

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
        filenames = getfilenamesfromdir(RAWDATAFOLDER, folder)
        for filename in filenames:
            if not check_consistent_activity(filename):
                isclear = False
        print("isClear: " + str(isclear))

def check_participants_id(folders):
    for folder in folders:
        print(f"{folder}:")
        hashmap = {}
        filenames = getfilenamesfromdir(PREPROCESSEDFOLDER2, folder)
        for filename in filenames:
            pid = re.split("P|A", filename)[1]
            if pid not in hashmap:
                hashmap[pid] = None
        print([k for k in hashmap.keys()])
        print("\n")

# For Preprocessed Data ------------------------------------------------------
# Store dataset to folder
# folders = ['1 December 2017 Dataset', '2 March 2017 Dataset', '3 June 2017 Dataset', '4 July 2018 Dataset', '5 February 2019 UoG Dataset', '6 February 2019 NG Homes Dataset', '7 March 2019 West Cumbria Dataset']
# for folder in folders:
#     print(folder + ":")
#     filenames = getfilenamesfromdir(RAWDATAFOLDER, folder)
#     for filename in filenames:
#         data = preprocess(folder, filename, False, False, True, True)
#         np.save(f"{PREPROCESSEDFOLDER}/{folder} Normalized/{filename[:-4]} Normalized.npy", data)
#         print(filename)
#     print("Done Saving!!")

def vstack_preprocessdata(folder, name):
    datalist = []
    filenames = getfilenamesfromdir(PREPROCESSEDFOLDER2, folder)
    for filename in filenames:
        # Repetition == 1
        if int(re.split("R| ", filename)[1].split('.')[0]) == 1:

        # Repetition == 2
        # if int(re.split("R| ", filename)[1].split('.')[0]) == 2:

        # Repetition == 2 & 3
        # if int(re.split("R| ", filename)[1].split('.')[0]) in (2, 3):
            print(filename + ": ")
            datalist.append(np.load(f"{PREPROCESSEDFOLDER}/{folder}/{filename}"))
    print("Storing...")
    print(np.array(datalist).shape)
    # Saved in activity data -----------------
    # np.save(f"{PREPROCESSEDFOLDER}/{name}.npy", np.array(datalist))
    # Saved in participant data --------------
    np.save(f"{PREPROCESSEDFOLDER2}/{name}.npy", np.array(datalist))
    print("Done Saving!!")

def vstack_datasets(folders, name):
    vstack_datasets = None
    for folder in folders:
        print(folder + ": ")
        if vstack_datasets is None:
            vstack_datasets = np.load(f"{PREPROCESSEDFOLDER2}/{folder} R1.npy")
        else:
            dataset = np.load(f"{PREPROCESSEDFOLDER2}/{folder} R1.npy")
            vstack_datasets = np.vstack((vstack_datasets, dataset))
    print("Storing...")
    print(vstack_datasets.shape)
    # Saved in activity data -----------------
    # np.save(f"{PREPROCESSEDFOLDER}/{name}.npy", vstack_datasets)
    # Saved in participant data --------------
    np.save(f"{PREPROCESSEDFOLDER2}/{name}.npy", vstack_datasets)
    print("Done Saving!!")

def store_activity_labels(folders):
    for folder in folders:
        activities_label = []
        print(folder + ":")
        filenames = getfilenamesfromdir(RAWDATAFOLDER, folder)
        for filename in filenames:
            # Repetition == 1
            # if int(re.split("R| ", filename)[1].split('.')[0]) == 1:
            activities_label.append(int(filename[0]) - 1)
        # np.save(f"{PREPROCESSEDFOLDER}/{folder[:-8]} Label R1.npy", np.array(activities_label))
        np.save(f"{PREPROCESSEDFOLDER}/{folder[:-8]} Label.npy", np.array(activities_label))
        print("Done Saving!!")

def store_participant_labels(folders, name):
    dataset1 = {"36":0, "37":1, "38":2, "39":3, "40":4, "41":5, "42":6, "43":7, "44":8, "45":9, "46":10, "47":11, "50":12, "51":13, "52":14, "53":15, "54":16, "55":17, "56":18, "57":19}
    dataset3 = {"14":20, "28":21, "29":22, "30":23, "31":24, "32":25, "33":26, "34":27}
    dataset4 = {"57":28, "58":29, "59":30, "60":31, "61":32, "62":33, "64":34, "65":35, "66":36, "67":37, "68":38, "69":39, "70":40, "71":41, "72":42}
    dataset5 = {"01":43, "02":44, "03":45, "04":46, "05":47, "06":48, "07":49, "08":50, "09":51, "10":52, "11":53, "12":54, "13":55, "14":56, "15":57, "16":58, "17":59}
    dataset6 = {"08":60}
    for folder in folders:
        activities_label = []
        print(folder + ":")
        filenames = getfilenamesfromdir(PREPROCESSEDFOLDER2, folder)
        for filename in filenames:
            # Repetition == 2 & 3
            if int(re.split("R| ", filename)[1].split('.')[0]) in (2, 3):
                activities_label.append(dataset1[re.split("P|A", filename)[1]])
        print(np.array(activities_label).shape)
        np.save(f"{PREPROCESSEDFOLDER2}/{name}.npy", np.array(activities_label))
        print("Done Saving!!")

def vstack_labels(folders, name):
    vstacklabel = None
    for folder in folders:
        print(folder + ": ")
        if vstacklabel is None:
            # vstacklabel = np.load(f"{PREPROCESSEDFOLDER}/{folder[:-8]} Label R1.npy")
            vstacklabel = np.load(f"{PREPROCESSEDFOLDER2}/{folder[:-8]} Label R1.npy")
        else:
            # label = np.load(f"{PREPROCESSEDFOLDER}/{folder[:-8]} Label R1.npy")
            label = np.load(f"{PREPROCESSEDFOLDER2}/{folder[:-8]} Label R1.npy")
            vstacklabel = np.concatenate((vstacklabel, label))
    print("Storing...")
    print(vstacklabel.shape)
    # Saved in activity data -----------------
    # np.save(f"{PREPROCESSEDFOLDER}/{name}.npy", vstacklabel)
    # Saved in participant data -----------------
    np.save(f"{PREPROCESSEDFOLDER2}/{name}.npy", vstacklabel)
    print("Done Saving!!")