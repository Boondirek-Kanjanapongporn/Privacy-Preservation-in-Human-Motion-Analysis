from preprocess.datapreprocessing import *
from helperfunctions import *
import tensorflow as tf

if __name__ == "__main__":
    PREPROCESSEDFOLDER = "D:/Individual Project/Preprocessed Activity Data"
    PREPROCESSEDFOLDER2 = "D:/Individual Project/Preprocessed Participant Data"
    folders = ['1 December 2017 Dataset', '2 March 2017 Dataset', '3 June 2017 Dataset', '4 July 2018 Dataset', '5 February 2019 UoG Dataset', '6 February 2019 NG Homes Dataset']
    folders_normalized = ['1 December 2017 Dataset Normalized', '2 March 2017 Dataset Normalized', '3 June 2017 Dataset Normalized', '4 July 2018 Dataset Normalized', '5 February 2019 UoG Dataset Normalized', '6 February 2019 NG Homes Dataset Normalized', '7 March 2019 West Cumbria Dataset Normalized']

    # Visualise Data Here: -----------------------------------------------------------------------------
    # preprocess('1 December 2017 Dataset', '1P36A01R01.dat', False, True, True, True)
    # preprocess('1 December 2017 Dataset', '2P36A02R01.dat', False, True, False, False)
    # preprocess('1 December 2017 Dataset', '3P36A03R01.dat', False, True, True, True)
    # preprocess('1 December 2017 Dataset', '4P36A04R01.dat', False, True, False, False)
    # preprocess('1 December 2017 Dataset', '5P36A05R01.dat', False, True, False, False)
    # preprocess('1 December 2017 Dataset', '6P36A06R01.dat', False, True, False, False)

    #----------------------------------------------------------------------------------------------------

    # Example Commands ------------------------------------
    # folders_normalized = ['1 December 2017 Dataset Normalized', '2 March 2017 Dataset Normalized', '3 June 2017 Dataset Normalized', '4 July 2018 Dataset Normalized', '5 February 2019 UoG Dataset Normalized', '6 February 2019 NG Homes Dataset Normalized', '7 March 2019 West Cumbria Dataset Normalized']
    # folders = ['1 December 2017 Dataset', '2 March 2017 Dataset', '3 June 2017 Dataset', '4 July 2018 Dataset', '5 February 2019 UoG Dataset', '6 February 2019 NG Homes Dataset', '7 March 2019 West Cumbria Dataset']
    # folders_label = ['1 December 2017 Label', '2 March 2017 Label', '3 June 2017 Label', '4 July 2018 Label', '5 February 2019 UoG Label', '6 February 2019 NG Homes Label', '7 March 2019 West Cumbria Label']
    # 1. preprocess('1 December 2017 Dataset', '1P36A01R01.dat', False, True, True, True)
    # 2. are_folders_clear(folders)
    # 3. vstack_preprocessdata('3 June 2017 Dataset Normalized', '3 June 2017 Dataset Normalized R1')
    # 4. vstack_datasets(folders_normalized, "dataset1to7 Normalized R1")
    # 5.2. store_activity_labels(folders)
    # 5.1. store_participant_labels(folders, '7 March 2019 West Cumbria Label R2&R3')
    # 6. vstack_labels(folders, "dataset1to7 Label")
    # Alternative:
    # 1. groupDataForParticipantRecognition1()
    # 2. groupDataForParticipantRecognition2()

    # Other commands ------------------------------------
    # Set default option of tensorflow to use GPU
    '''
    print(tf.test.gpu_device_name())
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print("Available GPUs:", physical_devices)

    # Set the GPU device to use (e.g., the first GPU)
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)  # Optionally, set memory growth
        tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
        print("Using GPU:", physical_devices[0])
    else:
        print("No GPUs available, using CPU.")
    '''

    # Store dataset to folder
    '''
    folders = ['1 December 2017 Dataset', '2 March 2017 Dataset', '3 June 2017 Dataset', '4 July 2018 Dataset', '5 February 2019 UoG Dataset', '6 February 2019 NG Homes Dataset', '7 March 2019 West Cumbria Dataset']
    for folder in folders:
        print(folder + ":")
        filenames = getfilenamesfromdir(folder)
        for filename in filenames:
            data = preprocess(folder, filename, False, False, True, True)
            np.save(f"{PREPROCESSEDFOLDER}/{folder} Normalized/{filename[:-4]} Normalized.npy", data)
            print(filename)
        print("Done Saving!!")
    '''

    