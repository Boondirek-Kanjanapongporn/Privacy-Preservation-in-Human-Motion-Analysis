from datapreprocessing import *
from helperfunctions import getfilenamesfromdir, are_folders_clear, store_dataset_by_dir, store_label_by_dir

if __name__ == "__main__":
    folders = ['7 March 2019 West Cumbria Dataset']
    

    # Example Commands
    # 1. folders = ['1 December 2017 Dataset', '2 March 2017 Dataset', '3 June 2017 Dataset', '4 July 2018 Dataset', '5 February 2019 UoG Dataset', '6 February 2019 NG Homes Dataset', '7 March 2019 West Cumbria Dataset']
    # 2. preprocess('6 February 2019 NG Homes Dataset', '1P20A01R1.dat', False, True, False)
    # 3. are_folders_clear(folders)
    # 4. store_dataset_by_dir(folders) # More recommended to do a folder at a time
    # 5. store_label_by_dir(folders)  # More recommended to do a folder at a time