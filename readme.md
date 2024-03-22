# Project Directory Structure
## Please Note that if you are clone from Github, you won't be getting the data and experimentedmodels as it is git ignored for confidentiality
## Please download the .zip folder provided in the submission!!
The most important directories in this project is `data/`, `src/`, and `experimentmodels/` directories:

1. `data/` directory:
This directory stores the essential data needed to train and evaluate the models and privacy-preservation mechanism
There are two important sub-directories:
* `processed/` directory:
        * This directory holds the processed versions of the radar data. The files within this directory have already been named and categorized following the data splitting strategies and the dataset type discussed in the dissertation.
        * Please do not change the name of the data files as it will break the possibility of running the evaluations
* `raw/1 December 2017 Dataset/` directory
        * The raw directory is actually supposed to hold all the radar dataset obtained from the University of Glasgow Radar dataset.
        * However, as it eats alot of space, I only retain the 6 radar files, 1 for each activity, for visualisation. This can be visualise in `src/main.py` but you will need to uncomment the code first to run (`In the Visualise Data Here` section).
        * If you want to get a hold of the full radar dataset you can go to `https://researchdata.gla.ac.uk/848/`.

2. `src/` directory:
        * This directory holds all the codes for running the evaluation and training of radar data
        * The important directory to be noted are `model/` and `privacy_preservation/` directories:

* `model/` directory:
        * This directory is quite big as it branches out to `activity recognition/`, `multitask recognition/`, and `participant recognition/` folders. 
        * In each of the subfolders, the most important ones to note are the `trainmodel.py` and `predictmodel.py` files. 
        * `trainmodel.py` file is use to train the model of that respective model type (for example, if it is in `activity recognition/` folder, then it is used to train activity classification models). 
        * Subsequently, `predictmodel.py` file is use to predict the trained model of that respective model type (for example, if it is in `participant recognition/` folder, then it is used to predict participant identification models)

* `privacy_preservation/` directory:
        * This is used to analyse the privacy-preserving mechanisms of the standard 𝜖-differential privacy and our proposed Adaptive Feature-based Perturbation (AFP) mechanism. The most important files to note is the `comparison_of_privacy_preservation_mechanisms.py` and `visualising_optimal_K_for_AFP.py` files which are used to visualisation the evaluation that we have gotten from conducting the research.
        * `comparison_of_privacy_preservation_mechanisms.py` file is used to compare the effectiveness of preserving privacy while retaining utility of both preservation mechanisms
        * `visualising_optimal_K_for_AFP.py` file shows the diagram of the preserving privacy and utility trend of K=100-1000 and K=1000-10000, to illustrate how we have concluded that K=1000 is the optimal value of K
        * The visualisations of these two files are also in the dissertation.

3. `experimentmodels/` directory:
        * This directory is to hold the finalized models trained during the hyperparameter tunings and ablation studies conducted in this project. 
        * Please note that the names of the model files are *Letter Sensitive* and please do not rename the files and the models contain here are used to load into the `predictmodel.py` files within `activity recognition/`, `multitask recognition/`, and `participant recognition/` folders and also in predict model files within `privacy_preservation/` folder.

## How to run training models and evaluation
All files are adjusted and configured to run relatively simply. You just have to navigate to the file you want to run and click run file to run that file. The files are run separately and individually. So you can run one at a time.

### 1. Model Train and Evaluation
* Navigate to the `activity recognition/`, `multitask recognition/`, or `participant recognition/` folders (whichever type of model you are interested in).
* Click run on the `trainmodel.py` file to run the training process of the model
* To evaluate the optimal model performance, you can simply click run on `predictmodel.py` to obtain the prediction plots and the confusion matrices

### 2. Privacy-preservation Mechanism Evaluation
* Navigate to the `privacy_preservation/` directory.
* Click run on the `comparison_of_privacy_preservation_mechanisms.py` file if you want to see the plot of comparing the effectiveness of preserving privacy while retaining utility of both preservation mechanisms.
* Click run on the `visualising_optimal_K_for_AFP.py` file to see the plot of the preserving privacy and utility trend of K=100-1000 and K=1000-10000 used to find the optimal value of K for AFP mechanism.



