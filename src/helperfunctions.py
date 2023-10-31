from os import listdir
from os.path import isfile, join
import numpy as np
import re

RAWDATAFOLDER = "D:/Individual Project/Data"
PREPROCESSEDFOLDER = "D:/Individual Project/Preprocessed Activity Data"
PREPROCESSEDFOLDER2 = "D:/Individual Project/Preprocessed Participant Data"

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

        # Repetition == 2
        if int(re.split("R| ", filename)[1].split('.')[0]) == 2:

        # Repetition == 1 & 3
        # if int(re.split("R| ", filename)[1].split('.')[0]) in (1, 3):
            print(filename + ": ")
            datalist.append(np.load(f"{PREPROCESSEDFOLDER}/{folder} Normalized/{filename[:-4]} Normalized.npy"))
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
            # vstack_datasets = np.load(f"{PREPROCESSEDFOLDER}/{folder}.npy")
            vstack_datasets = np.load(f"{PREPROCESSEDFOLDER2}/{folder} R2.npy")
        else:
            # dataset = np.load(f"{PREPROCESSEDFOLDER}/{folder}.npy")
            dataset = np.load(f"{PREPROCESSEDFOLDER2}/{folder} R2.npy")
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
        filenames = getfilenamesfromdir(folder)
        for filename in filenames:
            # Repetition == 1
            # if int(re.split("R| ", filename)[1].split('.')[0]) == 1:
            activities_label.append(int(filename[0]) - 1)
        # np.save(f"{PREPROCESSEDFOLDER}/{folder[:-8]} Label R1.npy", np.array(activities_label))
        np.save(f"{PREPROCESSEDFOLDER}/{folder[:-8]} Label.npy", np.array(activities_label))
        print("Done Saving!!")

def store_participant_labels(folders):
    dataset1 = {"36":0, "37":1, "38":2, "39":3, "40":4, "41":5, "42":6, "43":7, "44":8, "45":9, "46":10, "47":11, "50":12, "51":13, "52":14, "53":15, "54":16, "55":17, "56":18, "57":19}
    dataset2 = {"03":20, "10":21, "11":22, "12":23}
    dataset3 = {"14":24, "28":25, "29":26, "30":27, "31":28, "32":29, "33":30, "34":31, "35":32}
    dataset4 = {"57":33, "58":34, "59":35, "60":36, "61":37, "62":38, "63":39, "64":40, "65":41, "66":42, "67":43, "68":44, "69":45, "70":46, "71":47, "72":48}
    dataset5 = {"01":49, "02":50, "03":51, "04":52, "05":53, "06":54, "07":55, "08":56, "09":57, "10":58, "11":59, "12":60, "13":61, "14":62, "15":63, "16":64, "17":65}
    dataset6 = {"08":66, "18":67, "19":68, "20":69, "21":70, "22":71, "23":72, "24":73, "25":74, "26":75, "27":76, "28":77, "29":78, "30":79, "31":80, "32":81, "33":82, "34":83, "35":84, "36":85}
    dataset7 = {"37":86, "38":87, "39":88, "40":89, "41":90, "42":91, "43":92, "44":93, "45":94, "46":95, "47":96, "48":97, "49":98, "50":99, "51":100, "52":101, "53": 102, "54": 103, "55": 104, "56": 105}
    for folder in folders:
        activities_label = []
        print(folder + ":")
        filenames = getfilenamesfromdir(folder)
        for filename in filenames:
            # Repetition == 2
            if int(re.split("R| ", filename)[1].split('.')[0]) == 2:
                activities_label.append(dataset7[re.split("P|A", filename)[1]])
        np.save(f"{PREPROCESSEDFOLDER2}/{folder[:-8]} Label R2.npy", np.array(activities_label))
        print("Done Saving!!")

def vstack_labels(folders, name):
    vstacklabel = None
    for folder in folders:
        print(folder + ": ")
        if vstacklabel is None:
            # vstacklabel = np.load(f"{PREPROCESSEDFOLDER}/{folder[:-8]} Label R1.npy")
            # vstacklabel = np.load(f"{PREPROCESSEDFOLDER}/{folder[:-8]} Label.npy")
            vstacklabel = np.load(f"{PREPROCESSEDFOLDER2}/{folder[:-8]} Label R2.npy")
        else:
            # label = np.load(f"{PREPROCESSEDFOLDER}/{folder[:-8]} Label R1.npy")
            label = np.load(f"{PREPROCESSEDFOLDER2}/{folder[:-8]} Label R2.npy")
            vstacklabel = np.concatenate((vstacklabel, label))
    print("Storing...")
    print(vstacklabel.shape)
    # Saved in activity data -----------------
    # np.save(f"{PREPROCESSEDFOLDER}/{name}.npy", vstacklabel)
    # Saved in participant data -----------------
    np.save(f"{PREPROCESSEDFOLDER2}/{name}.npy", vstacklabel)
    print("Done Saving!!")