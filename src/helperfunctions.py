from os import listdir
from os.path import isfile, join
import numpy as np
import re

RAWDATAFOLDER = "D:/Individual Project/Data"
PREPROCESSEDFOLDER = "D:/Individual Project/Preprocessed Activity Data"
PREPROCESSEDFOLDER2 = "D:/Individual Project/Preprocessed Participant Data"
PREPROCESSEDFOLDER3 = "D:/Individual Project/Preprocessed Multitask Data"

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
    filenames = getfilenamesfromdir(PREPROCESSEDFOLDER, folder)
    # filenames = getfilenamesfromdir(PREPROCESSEDFOLDER2, folder)
    for filename in filenames:
        # Repetition == 1
        # if int(re.split("R| ", filename)[1].split('.')[0]) == 1:

        # Repetition == 2
        # if int(re.split("R| ", filename)[1].split('.')[0]) == 2:

        # Repetition == 2 & 3
        # if int(re.split("R| ", filename)[1].split('.')[0]) in (2, 3):
            print(filename + ": ")
            datalist.append(np.load(f"{PREPROCESSEDFOLDER}/{folder}/{filename}"))
    print("Storing...")
    print(np.array(datalist).shape)
    # Saved in activity data -----------------
    np.save(f"{PREPROCESSEDFOLDER}/{name}.npy", np.array(datalist))
    # Saved in participant data --------------
    # np.save(f"{PREPROCESSEDFOLDER2}/{name}.npy", np.array(datalist))
    print("Done Saving!!")

def vstack_datasets(folders, name):
    vstack_datasets = None
    for folder in folders:
        print(folder + ": ")
        if vstack_datasets is None:
            vstack_datasets = np.load(f"{PREPROCESSEDFOLDER}/{folder}.npy")
            # vstack_datasets = np.load(f"{PREPROCESSEDFOLDER2}/{folder} R2&R3.npy")
        else:
            dataset = np.load(f"{PREPROCESSEDFOLDER}/{folder}.npy")
            # dataset = np.load(f"{PREPROCESSEDFOLDER2}/{folder} R2&R3.npy")
            vstack_datasets = np.vstack((vstack_datasets, dataset))
    print("Storing...")
    print(vstack_datasets.shape)
    # Saved in activity data -----------------
    np.save(f"{PREPROCESSEDFOLDER}/{name}.npy", vstack_datasets)
    # Saved in participant data --------------
    # np.save(f"{PREPROCESSEDFOLDER2}/{name}.npy", vstack_datasets)
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
            # Repetition == 1
            if int(re.split("R| ", filename)[1].split('.')[0]) == 1:
            # Repetition == 2 & 3
            # if int(re.split("R| ", filename)[1].split('.')[0]) in (2, 3):
                activities_label.append(dataset6[re.split("P|A", filename)[1]])
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

def groupDataForParticipantRecognition1():
    testFileDataset = [3, 3, 2, 11, 15, 9, 11, 2, 7, 2, 7, 16, 8, 0, 1, 12, 5, 13, 13, 2, 10, 17, 17, 11, 4, 15, 12, 4, 2, 5, 6, 15, 5, 8, 6, 11, 11, 7, 4, 10, 8, 5, 2, 12, 6, 13, 0, 10, 12, 15, 4, 9, 12, 4, 13, 1, 11, 0, 7, 1, 10]
    validationParticipantID = [3, 8, 12, 15, 22, 28, 31, 32, 39, 47, 52, 54, 55, 57, 58]
    validationFileDataset = [12, 2, 5, 10, 0, 11, 0, 10, 4, 16, 15, 17, 3, 6, 13]
    trainDataset = []
    trainLabel = []
    validationDataset = []
    validationLabel = []
    testDataset = []
    testLabel = []
    validationptr = 0
    print("Grouping...")
    for i in range(61):
        print(f"Participant ID: {i}")
        inValidation = False
        if i in validationParticipantID:
            inValidation = True
        filenames = getfilenamesfromdir(f"{PREPROCESSEDFOLDER2}/Participants ID", i)
        for j, filename in enumerate(filenames):
            dataset = np.load(f"{PREPROCESSEDFOLDER2}/Participants ID/{i}/{filename}")
            if j == testFileDataset[i]:
                testLabel.append(i)
                testDataset.append(dataset)
            elif inValidation and j == validationFileDataset[validationptr]:
                print(f"ptr: {validationptr}")
                validationptr += 1
                inValidation = False
                validationLabel.append(i)
                validationDataset.append(dataset)
            else:
                trainLabel.append(i)
                trainDataset.append(dataset)

    trainDataset = np.array(trainDataset)
    trainLabel = np.array(trainLabel)
    validationDataset = np.array(validationDataset)
    validationLabel = np.array(validationLabel)
    testDataset = np.array(testDataset)
    testLabel = np.array(testLabel)
    
    print("Storing...")
    print("Train Dataset")
    print(f"Dataset: {trainDataset.shape}")
    print(f"Label: {trainLabel.shape}")
    np.save(f"{PREPROCESSEDFOLDER2}/trainDataset1.npy", trainDataset)
    np.save(f"{PREPROCESSEDFOLDER2}/trainLabel1.npy", trainLabel)

    print("\nValidation Dataset")
    print(f"Dataset: {validationDataset.shape}")
    print(f"Label: {validationLabel.shape}")
    np.save(f"{PREPROCESSEDFOLDER2}/validationDataset1.npy", validationDataset)
    np.save(f"{PREPROCESSEDFOLDER2}/validationLabel1.npy", validationLabel)

    print("\nTest Dataset")
    print(f"Dataset: {testDataset.shape}")
    print(f"Label: {testLabel.shape}")
    np.save(f"{PREPROCESSEDFOLDER2}/testDataset1.npy", testDataset)
    np.save(f"{PREPROCESSEDFOLDER2}/testLabel1.npy", testLabel)
    print("Done Saving!!")

def groupDataForParticipantRecognition2():
    testFileDataset = [6, 3, 17, 8, 6, 5, 8, 0, 15, 8, 6, 16, 13, 3, 17, 15, 0, 10, 3, 17, 14, 1, 4, 16, 2, 16, 12, 11, 12, 0, 2, 16, 7, 4, 8, 1, 7, 8, 2, 2, 16, 12, 2, 1, 5, 6, 0, 12, 9, 14, 4, 10, 13, 14, 13, 8, 1, 0, 14, 7, 0]
    validationFileDataset = [[0, 7, 12, 13], [1, 2, 8, 13], [3, 5, 12, 16], [0, 4, 6, 16], [1, 13, 14, 15], [1, 10, 14, 17], [2, 9, 12, 15], [8, 9, 13, 17], [0, 6, 8, 13], [1, 4, 7, 13], [3, 10, 14, 15], [0, 1, 4, 5], [1, 2, 8, 11], [0, 6, 9, 16], [0, 6, 13, 16], [4, 8, 11, 12], [4, 6, 7, 8], [4, 6, 15, 16], [6, 12, 13, 16], [1, 2, 3, 7], [11, 13, 15, 16], [3, 10, 11, 13], [1, 7, 11, 16], [6, 9, 13, 15], [0, 4, 10, 12], [2, 8, 10, 12], [1, 7, 10, 14], [0, 2, 6, 14], [0, 4, 7, 14], [9, 10, 11, 17], [5, 12, 15, 17], [0, 1, 3, 10], [1, 8, 13, 17], [0, 2, 8, 13], [4, 7, 9, 11], [9, 10, 15, 16], [4, 10, 15, 17], [3, 9, 12, 13], [5, 9, 13, 17], [1, 5, 10, 14], [4, 6, 12, 15], [2, 6, 13, 17], [4, 7, 13, 17], [0, 4, 12, 16], [0, 3, 6, 13], [3, 9, 13, 14], [1, 3, 8, 10], [4, 5, 14, 15], [2, 10, 12, 16], [0, 3, 9, 17], [0, 13, 14, 16], [3, 7, 12, 17], [3, 9, 16, 17], [3, 4, 9, 10], [1, 9, 10, 17], [3, 7, 13, 15], [2, 4, 9, 13], [5, 7, 15, 16], [7, 8, 11, 17], [8, 14, 15, 17], [2, 4, 8, 13]]
    trainDataset = []
    trainLabel = []
    validationDataset = []
    validationLabel = []
    testDataset = []
    testLabel = []
    print("Grouping...")
    for i in range(61):
        print(f"Participant ID: {i}")
        filenames = getfilenamesfromdir(f"{PREPROCESSEDFOLDER2}/Participants ID", i)
        for j, filename in enumerate(filenames):
            dataset = np.load(f"{PREPROCESSEDFOLDER2}/Participants ID/{i}/{filename}")
            if j == testFileDataset[i]:
                testLabel.append(i)
                testDataset.append(dataset)
            elif j in validationFileDataset[i]:
                validationLabel.append(i)
                validationDataset.append(dataset)
            else:
                trainLabel.append(i)
                trainDataset.append(dataset)

    trainDataset = np.array(trainDataset)
    trainLabel = np.array(trainLabel)
    validationDataset = np.array(validationDataset)
    validationLabel = np.array(validationLabel)
    testDataset = np.array(testDataset)
    testLabel = np.array(testLabel)
    
    print("Storing...")
    print("Train Dataset")
    print(f"Dataset: {trainDataset.shape}")
    print(f"Label: {trainLabel.shape}")
    np.save(f"{PREPROCESSEDFOLDER2}/trainDataset.npy", trainDataset)
    np.save(f"{PREPROCESSEDFOLDER2}/trainLabel.npy", trainLabel)

    print("\nValidation Dataset")
    print(f"Dataset: {validationDataset.shape}")
    print(f"Label: {validationLabel.shape}")
    np.save(f"{PREPROCESSEDFOLDER2}/validationDataset.npy", validationDataset)
    np.save(f"{PREPROCESSEDFOLDER2}/validationLabel.npy", validationLabel)

    print("\nTest Dataset")
    print(f"Dataset: {testDataset.shape}")
    print(f"Label: {testLabel.shape}")
    np.save(f"{PREPROCESSEDFOLDER2}/testDataset.npy", testDataset)
    np.save(f"{PREPROCESSEDFOLDER2}/testLabel.npy", testLabel)
    print("Done Saving!!")

def groupDataForParticipantRecognitionByActivity():
    participantActivityDataset = [[[0, 1, 2], [3, 4, 5], [8, 6, 7], [10, 9, 11], [13, 12, 14], [17, 15, 16]], [[1, 2, 0], [4, 5, 3], [6, 8, 7], [9, 10, 11], [14, 13, 12], [15, 16, 17]], [[2, 0, 1], [3, 5, 4], [8, 7, 6], [9, 11, 10], [14, 13, 12], [15, 16, 17]], [[2, 1, 0], [4, 3, 5], [8, 6, 7], [9, 11, 10], [14, 12, 13], [16, 17, 15]], [[2, 1, 0], [3, 5, 4], [8, 6, 7], [9, 10, 11], [12, 14, 13], [16, 15, 17]], [[1, 0, 2], [3, 4, 5], [8, 6, 7], [11, 9, 10], [13, 14, 12], [16, 15, 17]], [[1, 0, 2], [4, 3, 5], [6, 7, 8], [9, 11, 10], [12, 14, 13], [17, 16, 15]], [[1, 0, 2], [3, 5, 4], [6, 7, 8], [10, 11, 9], [13, 14, 12], [15, 16, 17]], [[2, 1, 0], [5, 4, 3], [8, 7, 6], [10, 9, 11], [14, 12, 13], [15, 17, 16]], [[1, 2, 0], [3, 5, 4], [7, 8, 6], [11, 10, 9], [13, 12, 14], [15, 17, 16]], [[2, 1, 0], [4, 5, 3], [8, 7, 6], [9, 11, 10], [13, 12, 14], [16, 15, 17]], [[2, 0, 1], [3, 4, 5], [7, 6, 8], [11, 10, 9], [14, 13, 12], [17, 15, 16]], [[2, 1, 0], [5, 3, 4], [7, 8, 6], [11, 10, 9], [12, 14, 13], [16, 17, 15]], [[1, 2, 0], [5, 3, 4], [8, 6, 7], [11, 10, 9], [14, 13, 12], [17, 16, 15]], [[1, 0, 2], [3, 5, 4], [8, 6, 7], [10, 11, 9], [14, 13, 12], [17, 16, 15]], [[0, 1, 2], [3, 5, 4], [6, 7, 8], [10, 9, 11], [12, 13, 14], [16, 15, 17]], [[2, 0, 1], [3, 4, 5], [6, 8, 7], [11, 10, 9], [12, 14, 13], [16, 15, 17]], [[1, 0, 2], [5, 3, 4], [6, 8, 7], [9, 10, 11], [14, 12, 13], [15, 16, 17]], [[2, 1, 0], [4, 3, 5], [7, 8, 6], [9, 10, 11], [13, 12, 14], [16, 17, 15]], [[1, 2, 0], [4, 3, 5], [7, 8, 6], [9, 10, 11], [12, 14, 13], [17, 16, 15]], [[0, 2, 1], [4, 5, 3], [8, 7, 6], [11, 9, 10], [14, 12, 13], [16, 15, 17]], [[1, 0, 2], [4, 3, 5], [8, 6, 7], [11, 9, 10], [14, 13, 12], [16, 17, 15]], [[2, 0, 1], [4, 5, 3], [6, 8, 7], [9, 11, 10], [12, 14, 13], [17, 16, 15]], [[2, 0, 1], [4, 3, 5], [6, 8, 7], [10, 9, 11], [13, 14, 12], [16, 15, 17]], [[2, 0, 1], [5, 3, 4], [8, 6, 7], [9, 10, 11], [14, 13, 12], [17, 15, 16]], [[2, 0, 1], [5, 3, 4], [7, 6, 8], [11, 10, 9], [12, 13, 14], [16, 17, 15]], [[0, 1, 2], [4, 3, 5], [6, 8, 7], [10, 9, 11], [14, 12, 13], [17, 15, 16]], [[1, 0, 2], [3, 4, 5], [6, 7, 8], [10, 11, 9], [12, 14, 13], [15, 17, 16]], [[2, 0, 1], [5, 3, 4], [8, 7, 6], [9, 10, 11], [12, 14, 13], [15, 16, 17]], [[1, 2, 0], [3, 5, 4], [7, 8, 6], [11, 10, 9], [13, 12, 14], [17, 15, 16]], [[1, 0, 2], [5, 3, 4], [8, 6, 7], [9, 11, 10], [12, 14, 13], [17, 16, 15]], [[1, 2, 0], [3, 5, 4], [8, 6, 7], [11, 10, 9], [12, 14, 13], [17, 15, 16]], [[1, 2, 0], [3, 4, 5], [6, 8, 7], [10, 11, 9], [12, 13, 14], [16, 15, 17]], [[1, 0, 2], [4, 5, 3], [8, 7, 6], [10, 11, 9], [14, 12, 13], [17, 16, 15]], [[2, 0, 1], [4, 3, 5], [6, 8, 7], [9, 10, 11], [13, 14, 12], [17, 16, 15]], [[1, 2, 0], [4, 3, 5], [7, 6, 8], [11, 10, 9], [13, 12, 14], [16, 17, 15]], [[0, 1, 2], [5, 4, 3], [6, 7, 8], [10, 9, 11], [12, 13, 14], [15, 16, 17]], [[2, 1, 0], [3, 5, 4], [7, 8, 6], [9, 10, 11], [14, 13, 12], [16, 17, 15]], [[0, 2, 1], [4, 5, 3], [8, 7, 6], [10, 11, 9], [12, 14, 13], [17, 16, 15]], [[0, 2, 1], [4, 5, 3], [7, 8, 6], [11, 10, 9], [13, 14, 12], [17, 16, 15]], [[0, 2, 1], [5, 3, 4], [7, 6, 8], [11, 10, 9], [14, 12, 13], [17, 16, 15]], [[1, 2, 0], [3, 4, 5], [7, 6, 8], [10, 11, 9], [13, 12, 14], [16, 17, 15]], [[2, 0, 1], [3, 5, 4], [8, 6, 7], [10, 11, 9], [14, 13, 12], [15, 16, 17]], [[2, 0, 1], [3, 4, 5], [8, 6, 7], [9, 11, 10], [12, 13, 14], [17, 16, 15]], [[1, 2, 0], [3, 5, 4], [8, 7, 6], [10, 11, 9], [13, 12, 14], [17, 16, 15]], [[2, 1, 0], [5, 3, 4], [7, 6, 8], [9, 10, 11], [14, 12, 13], [15, 16, 17]], [[2, 1, 0], [5, 4, 3], [6, 8, 7], [11, 10, 9], [12, 13, 14], [16, 15, 17]], [[1, 2, 0], [4, 3, 5], [6, 8, 7], [11, 9, 10], [14, 13, 12], [15, 17, 16]], [[0, 1, 2], [5, 4, 3], [7, 6, 8], [9, 11, 10], [12, 14, 13], [16, 17, 15]], [[2, 0, 1], [3, 4, 5], [7, 8, 6], [9, 11, 10], [12, 13, 14], [16, 15, 17]], [[0, 2, 1], [5, 3, 4], [6, 8, 7], [9, 10, 11], [12, 14, 13], [15, 17, 16]], [[1, 0, 2], [3, 5, 4], [8, 6, 7], [10, 11, 9], [14, 12, 13], [17, 15, 16]], [[0, 1, 2], [4, 5, 3], [7, 8, 6], [9, 10, 11], [14, 13, 12], [17, 15, 16]], [[1, 2, 0], [5, 4, 3], [8, 6, 7], [9, 11, 10], [14, 13, 12], [15, 16, 17]], [[0, 1, 2], [3, 5, 4], [6, 8, 7], [11, 10, 9], [13, 14, 12], [17, 15, 16]], [[2, 1, 0], [3, 4, 5], [8, 6, 7], [9, 10, 11], [14, 12, 13], [17, 15, 16]], [[1, 0, 2], [3, 4, 5], [8, 6, 7], [10, 11, 9], [14, 12, 13], [16, 17, 15]], [[2, 1, 0], [5, 4, 3], [7, 6, 8], [11, 9, 10], [14, 12, 13], [15, 17, 16]], [[1, 2, 0], [5, 4, 3], [6, 8, 7], [9, 11, 10], [13, 14, 12], [17, 15, 16]], [[2, 1, 0], [3, 5, 4], [8, 7, 6], [10, 9, 11], [12, 14, 13], [17, 15, 16]], [[1, 0, 2], [3, 5, 4], [6, 7, 8], [9, 11, 10], [13, 12, 14], [17, 16, 15]]]
    trainDataset_walk = []
    validationDataset_walk = []
    testDataset_walk = []
    trainDataset_sit = []
    validationDataset_sit = []
    testDataset_sit = []
    trainDataset_standup = []
    validationDataset_standup = []
    testDataset_standup = []
    trainDataset_pickup = []
    validationDataset_pickup = []
    testDataset_pickup = []
    trainDataset_drink = []
    validationDataset_drink = []
    testDataset_drink = []
    trainDataset_fall = []
    validationDataset_fall = []
    testDataset_fall = []
    datasetLabel = []
    print("Grouping...")
    for i, participant in enumerate(participantActivityDataset):
        print(f"Participant ID: {i}")
        datasetLabel.append(i)
        filenames = getfilenamesfromdir(f"{PREPROCESSEDFOLDER2}/Participants ID", i)
        for j, filename in enumerate(filenames):
            dataset = np.load(f"{PREPROCESSEDFOLDER2}/Participants ID/{i}/{filename}")
            acitivityCategory = j // 3
            idx = participant[acitivityCategory].index(j)
            if acitivityCategory == 0:
                if idx == 0:
                    trainDataset_walk.append(dataset)
                elif idx == 1:
                    validationDataset_walk.append(dataset)
                else:
                    testDataset_walk.append(dataset)
            elif acitivityCategory == 1:
                if idx == 0:
                    trainDataset_sit.append(dataset)
                elif idx == 1:
                    validationDataset_sit.append(dataset)
                else:
                    testDataset_sit.append(dataset)
            elif acitivityCategory == 2:
                if idx == 0:
                    trainDataset_standup.append(dataset)
                elif idx == 1:
                    validationDataset_standup.append(dataset)
                else:
                    testDataset_standup.append(dataset)
            elif acitivityCategory == 3:
                if idx == 0:
                    trainDataset_pickup.append(dataset)
                elif idx == 1:
                    validationDataset_pickup.append(dataset)
                else:
                    testDataset_pickup.append(dataset)
            elif acitivityCategory == 4:
                if idx == 0:
                    trainDataset_drink.append(dataset)
                elif idx == 1:
                    validationDataset_drink.append(dataset)
                else:
                    testDataset_drink.append(dataset)
            else:
                if idx == 0:
                    trainDataset_fall.append(dataset)
                elif idx == 1:
                    validationDataset_fall.append(dataset)
                else:
                    testDataset_fall.append(dataset)

    trainDataset_walk = np.array(trainDataset_walk)
    validationDataset_walk = np.array(validationDataset_walk)
    testDataset_walk = np.array(testDataset_walk)

    trainDataset_sit = np.array(trainDataset_sit)
    validationDataset_sit = np.array(validationDataset_sit)
    testDataset_sit = np.array(testDataset_sit)

    trainDataset_standup = np.array(trainDataset_standup)
    validationDataset_standup = np.array(validationDataset_standup)
    testDataset_standup = np.array(testDataset_standup)

    trainDataset_pickup = np.array(trainDataset_pickup)
    validationDataset_pickup = np.array(validationDataset_pickup)
    testDataset_pickup = np.array(testDataset_pickup)

    trainDataset_drink = np.array(trainDataset_drink)
    validationDataset_drink = np.array(validationDataset_drink)
    testDataset_drink = np.array(testDataset_drink)

    trainDataset_fall = np.array(trainDataset_fall)
    validationDataset_fall = np.array(validationDataset_fall)
    testDataset_fall = np.array(testDataset_fall)

    datasetLabel = np.array(datasetLabel)
    
    print("Storing...")
    print("1. Walk")
    print("Train Dataset")
    print(f"Dataset: {trainDataset_walk.shape}")
    np.save(f"{PREPROCESSEDFOLDER2}/trainDataset_walk.npy", trainDataset_walk)
    print("Validation Dataset")
    print(f"Dataset: {validationDataset_walk.shape}")
    np.save(f"{PREPROCESSEDFOLDER2}/validationDataset_walk.npy", validationDataset_walk)
    print("Test Dataset")
    print(f"Dataset: {testDataset_walk.shape}")
    np.save(f"{PREPROCESSEDFOLDER2}/testDataset_walk.npy", testDataset_walk)

    print("\n2. Sit")
    print("Train Dataset")
    print(f"Dataset: {trainDataset_sit.shape}")
    np.save(f"{PREPROCESSEDFOLDER2}/trainDataset_sit.npy", trainDataset_sit)
    print("Validation Dataset")
    print(f"Dataset: {validationDataset_sit.shape}")
    np.save(f"{PREPROCESSEDFOLDER2}/validationDataset_sit.npy", validationDataset_sit)
    print("Test Dataset")
    print(f"Dataset: {testDataset_sit.shape}")
    np.save(f"{PREPROCESSEDFOLDER2}/testDataset_sit.npy", testDataset_sit)

    print("\n3. Standup")
    print("Train Dataset")
    print(f"Dataset: {trainDataset_standup.shape}")
    np.save(f"{PREPROCESSEDFOLDER2}/trainDataset_standup.npy", trainDataset_standup)
    print("Validation Dataset")
    print(f"Dataset: {validationDataset_standup.shape}")
    np.save(f"{PREPROCESSEDFOLDER2}/validationDataset_standup.npy", validationDataset_standup)
    print("Test Dataset")
    print(f"Dataset: {testDataset_standup.shape}")
    np.save(f"{PREPROCESSEDFOLDER2}/testDataset_standup.npy", testDataset_standup)

    print("\n4. Pickup")
    print("Train Dataset")
    print(f"Dataset: {trainDataset_pickup.shape}")
    np.save(f"{PREPROCESSEDFOLDER2}/trainDataset_pickup.npy", trainDataset_pickup)
    print("Validation Dataset")
    print(f"Dataset: {validationDataset_pickup.shape}")
    np.save(f"{PREPROCESSEDFOLDER2}/validationDataset_pickup.npy", validationDataset_pickup)
    print("Test Dataset")
    print(f"Dataset: {testDataset_pickup.shape}")
    np.save(f"{PREPROCESSEDFOLDER2}/testDataset_pickup.npy", testDataset_pickup)

    print("\n5. Drink")
    print("Train Dataset")
    print(f"Dataset: {trainDataset_drink.shape}")
    np.save(f"{PREPROCESSEDFOLDER2}/trainDataset_drink.npy", trainDataset_drink)
    print("Validation Dataset")
    print(f"Dataset: {validationDataset_drink.shape}")
    np.save(f"{PREPROCESSEDFOLDER2}/validationDataset_drink.npy", validationDataset_drink)
    print("Test Dataset")
    print(f"Dataset: {testDataset_drink.shape}")
    np.save(f"{PREPROCESSEDFOLDER2}/testDataset_drink.npy", testDataset_drink)

    print("\n6. Fall")
    print("Train Dataset")
    print(f"Dataset: {trainDataset_fall.shape}")
    np.save(f"{PREPROCESSEDFOLDER2}/trainDataset_fall.npy", trainDataset_fall)
    print("Validation Dataset")
    print(f"Dataset: {validationDataset_fall.shape}")
    np.save(f"{PREPROCESSEDFOLDER2}/validationDataset_fall.npy", validationDataset_fall)
    print("Test Dataset")
    print(f"Dataset: {testDataset_fall.shape}")
    np.save(f"{PREPROCESSEDFOLDER2}/testDataset_fall.npy", testDataset_fall)

    print("\nActivity label")
    print(f"Label: {datasetLabel.shape}")
    np.save(f"{PREPROCESSEDFOLDER2}/datasetLabel.npy", datasetLabel)
    print("Done Saving!!")

def groupDataForMultitaskRecognition():
    testFileDataset = [6, 3, 17, 8, 6, 5, 8, 0, 15, 8, 6, 16, 13, 3, 17, 15, 0, 10, 3, 17, 14, 1, 4, 16, 2, 16, 12, 11, 12, 0, 2, 16, 7, 4, 8, 1, 7, 8, 2, 2, 16, 12, 2, 1, 5, 6, 0, 12, 9, 14, 4, 10, 13, 14, 13, 8, 1, 0, 14, 7, 0]
    validationFileDataset = [[0, 7, 12, 13], [1, 2, 8, 13], [3, 5, 12, 16], [0, 4, 6, 16], [1, 13, 14, 15], [1, 10, 14, 17], [2, 9, 12, 15], [8, 9, 13, 17], [0, 6, 8, 13], [1, 4, 7, 13], [3, 10, 14, 15], [0, 1, 4, 5], [1, 2, 8, 11], [0, 6, 9, 16], [0, 6, 13, 16], [4, 8, 11, 12], [4, 6, 7, 8], [4, 6, 15, 16], [6, 12, 13, 16], [1, 2, 3, 7], [11, 13, 15, 16], [3, 10, 11, 13], [1, 7, 11, 16], [6, 9, 13, 15], [0, 4, 10, 12], [2, 8, 10, 12], [1, 7, 10, 14], [0, 2, 6, 14], [0, 4, 7, 14], [9, 10, 11, 17], [5, 12, 15, 17], [0, 1, 3, 10], [1, 8, 13, 17], [0, 2, 8, 13], [4, 7, 9, 11], [9, 10, 15, 16], [4, 10, 15, 17], [3, 9, 12, 13], [5, 9, 13, 17], [1, 5, 10, 14], [4, 6, 12, 15], [2, 6, 13, 17], [4, 7, 13, 17], [0, 4, 12, 16], [0, 3, 6, 13], [3, 9, 13, 14], [1, 3, 8, 10], [4, 5, 14, 15], [2, 10, 12, 16], [0, 3, 9, 17], [0, 13, 14, 16], [3, 7, 12, 17], [3, 9, 16, 17], [3, 4, 9, 10], [1, 9, 10, 17], [3, 7, 13, 15], [2, 4, 9, 13], [5, 7, 15, 16], [7, 8, 11, 17], [8, 14, 15, 17], [2, 4, 8, 13]]
    trainDataset = []
    trainLabel_participant = []
    trainLabel_activity = []
    validationDataset = []
    validationLabel_participant = []
    validationLabel_activity = []
    testDataset = []
    testLabel_participant = []
    testLabel_activity = []
    print("Grouping...")
    for i in range(61):
        print(f"Participant ID: {i}")
        filenames = getfilenamesfromdir(f"{PREPROCESSEDFOLDER3}/Participants ID", i)
        for j, filename in enumerate(filenames):
            dataset = np.load(f"{PREPROCESSEDFOLDER3}/Participants ID/{i}/{filename}")
            if j == testFileDataset[i]:
                testLabel_participant.append(i)
                testLabel_activity.append(int(filename[0]) - 1)
                testDataset.append(dataset)
            elif j in validationFileDataset[i]:
                validationLabel_participant.append(i)
                validationLabel_activity.append(int(filename[0]) - 1)
                validationDataset.append(dataset)
            else:
                trainLabel_participant.append(i)
                trainLabel_activity.append(int(filename[0]) - 1)
                trainDataset.append(dataset)

    trainDataset = np.array(trainDataset)
    trainLabel_participant = np.array(trainLabel_participant)
    trainLabel_activity = np.array(trainLabel_activity)
    validationDataset = np.array(validationDataset)
    validationLabel_participant = np.array(validationLabel_participant)
    validationLabel_activity = np.array(validationLabel_activity)
    testDataset = np.array(testDataset)
    testLabel_participant = np.array(testLabel_participant)
    testLabel_activity = np.array(testLabel_activity)
    
    print("Storing...")
    print("Train Dataset")
    print(f"Dataset: {trainDataset.shape}")
    print(f"Activity Label: {trainLabel_activity.shape}")
    print(f"Participant Label: {trainLabel_participant.shape}")
    np.save(f"{PREPROCESSEDFOLDER3}/trainDataset.npy", trainDataset)
    np.save(f"{PREPROCESSEDFOLDER3}/trainLabel_activity.npy", trainLabel_activity)
    np.save(f"{PREPROCESSEDFOLDER3}/trainLabel_participant.npy", trainLabel_participant)

    print("\nValidation Dataset")
    print(f"Dataset: {validationDataset.shape}")
    print(f"Activity Label: {validationLabel_activity.shape}")
    print(f"Participant Label: {validationLabel_participant.shape}")
    np.save(f"{PREPROCESSEDFOLDER3}/validationDataset.npy", validationDataset)
    np.save(f"{PREPROCESSEDFOLDER3}/validationLabel_activity.npy", validationLabel_activity)
    np.save(f"{PREPROCESSEDFOLDER3}/validationLabel_participant.npy", validationLabel_participant)

    print("\nTest Dataset")
    print(f"Dataset: {testDataset.shape}")
    print(f"Activity Label: {testLabel_activity.shape}")
    print(f"Participant Label: {testLabel_participant.shape}")
    np.save(f"{PREPROCESSEDFOLDER3}/testDataset.npy", testDataset)
    np.save(f"{PREPROCESSEDFOLDER3}/testLabel_activity.npy", testLabel_activity)
    np.save(f"{PREPROCESSEDFOLDER3}/testLabel_participant.npy", testLabel_participant)
    print("Done Saving!!")

def groupDataForMultitaskRecognition30():
    testFileDataset = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 16, 12, 11, 12, 0, 2, 16, 7, 4, 8, 1, 7, 8, 2, 2, 16, 12, 2, 1, 5, 6, 0, 12, 9, 14, 4, 10, 13, 14, 13, None, None, None, None, None, None]
    validationFileDataset = [[0, 7, 12, 13], [1, 2, 8, 13], [3, 5, 12, 16], [0, 4, 6, 16], [1, 13, 14, 15], [1, 10, 14, 17], [2, 9, 12, 15], [8, 9, 13, 17], [0, 6, 8, 13], [1, 4, 7, 13], [3, 10, 14, 15], [0, 1, 4, 5], [1, 2, 8, 11], [0, 6, 9, 16], [0, 6, 13, 16], [4, 8, 11, 12], [4, 6, 7, 8], [4, 6, 15, 16], [6, 12, 13, 16], [1, 2, 3, 7], [11, 13, 15, 16], [3, 10, 11, 13], [1, 7, 11, 16], [6, 9, 13, 15], [0, 4, 10, 12], [2, 8, 10, 12], [1, 7, 10, 14], [0, 2, 6, 14], [0, 4, 7, 14], [9, 10, 11, 17], [5, 12, 15, 17], [0, 1, 3, 10], [1, 8, 13, 17], [0, 2, 8, 13], [4, 7, 9, 11], [9, 10, 15, 16], [4, 10, 15, 17], [3, 9, 12, 13], [5, 9, 13, 17], [1, 5, 10, 14], [4, 6, 12, 15], [2, 6, 13, 17], [4, 7, 13, 17], [0, 4, 12, 16], [0, 3, 6, 13], [3, 9, 13, 14], [1, 3, 8, 10], [4, 5, 14, 15], [2, 10, 12, 16], [0, 3, 9, 17], [0, 13, 14, 16], [3, 7, 12, 17], [3, 9, 16, 17], [3, 4, 9, 10], [1, 9, 10, 17], [3, 7, 13, 15], [2, 4, 9, 13], [5, 7, 15, 16], [7, 8, 11, 17], [8, 14, 15, 17], [2, 4, 8, 13]]
    trainDataset = []
    trainLabel_participant = []
    trainLabel_activity = []
    validationDataset = []
    validationLabel_participant = []
    validationLabel_activity = []
    testDataset = []
    testLabel_participant = []
    testLabel_activity = []
    labelID = 0
    print("Grouping...")
    for i in range(61):
        if testFileDataset[i] == None:
            continue
        print(f"Participant ID: {i}, labelID: {labelID}")
        filenames = getfilenamesfromdir(f"{PREPROCESSEDFOLDER3}/Participants ID", i)
        for j, filename in enumerate(filenames):
            dataset = np.load(f"{PREPROCESSEDFOLDER3}/Participants ID/{i}/{filename}")
            if j == testFileDataset[i]:
                testLabel_participant.append(labelID)
                testLabel_activity.append(int(filename[0]) - 1)
                testDataset.append(dataset)
            elif j in validationFileDataset[i]:
                validationLabel_participant.append(labelID)
                validationLabel_activity.append(int(filename[0]) - 1)
                validationDataset.append(dataset)
            else:
                trainLabel_participant.append(labelID)
                trainLabel_activity.append(int(filename[0]) - 1)
                trainDataset.append(dataset)
        labelID += 1

    trainDataset = np.array(trainDataset)
    trainLabel_participant = np.array(trainLabel_participant)
    trainLabel_activity = np.array(trainLabel_activity)
    validationDataset = np.array(validationDataset)
    validationLabel_participant = np.array(validationLabel_participant)
    validationLabel_activity = np.array(validationLabel_activity)
    testDataset = np.array(testDataset)
    testLabel_participant = np.array(testLabel_participant)
    testLabel_activity = np.array(testLabel_activity)
    
    print("Storing...")
    print("Train Dataset")
    print(f"Dataset: {trainDataset.shape}")
    print(f"Activity Label: {trainLabel_activity.shape}")
    print(f"Participant Label: {trainLabel_participant.shape}")
    np.save(f"{PREPROCESSEDFOLDER3}/trainDataset30.npy", trainDataset)
    np.save(f"{PREPROCESSEDFOLDER3}/trainLabel_activity30.npy", trainLabel_activity)
    np.save(f"{PREPROCESSEDFOLDER3}/trainLabel_participant30.npy", trainLabel_participant)

    print("\nValidation Dataset")
    print(f"Dataset: {validationDataset.shape}")
    print(f"Activity Label: {validationLabel_activity.shape}")
    print(f"Participant Label: {validationLabel_participant.shape}")
    np.save(f"{PREPROCESSEDFOLDER3}/validationDataset30.npy", validationDataset)
    np.save(f"{PREPROCESSEDFOLDER3}/validationLabel_activity30.npy", validationLabel_activity)
    np.save(f"{PREPROCESSEDFOLDER3}/validationLabel_participant30.npy", validationLabel_participant)

    print("\nTest Dataset")
    print(f"Dataset: {testDataset.shape}")
    print(f"Activity Label: {testLabel_activity.shape}")
    print(f"Participant Label: {testLabel_participant.shape}")
    np.save(f"{PREPROCESSEDFOLDER3}/testDataset30.npy", testDataset)
    np.save(f"{PREPROCESSEDFOLDER3}/testLabel_activity30.npy", testLabel_activity)
    np.save(f"{PREPROCESSEDFOLDER3}/testLabel_participant30.npy", testLabel_participant)
    print("Done Saving!!")

