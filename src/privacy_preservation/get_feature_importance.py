from sklearn.ensemble import RandomForestClassifier
import numpy as np
import tensorflow as tf
from pathlib import Path

# PREPROCESSEDFOLDER = "D:/Individual Project/Preprocessed Multitask Data"
PREPROCESSEDFOLDER = "../../data/processed"

TESTDATASET_FILE = "testDataset30.npy"
LABEL_FILE_ACTIVITY = "testLabel_activity30.npy"
LABEL_FILE_PARTICIPANT = "testLabel_participant30.npy"
TRAINED_MODEL = "experimentmodels/multitask_recognition_cnn5_97_70.h5"

script_location = Path(__file__).resolve().parent
data_folder = script_location / PREPROCESSEDFOLDER
data_path = data_folder / TESTDATASET_FILE
labels_path_activity = data_folder / LABEL_FILE_ACTIVITY 
labels_path_participant = data_folder / LABEL_FILE_PARTICIPANT

# Load data and labels
# test_data = np.load(f"{PREPROCESSEDFOLDER}/{TESTDATASET_FILE}")
# labels_activity = np.load(f"{PREPROCESSEDFOLDER}/{LABEL_FILE_ACTIVITY}")
# labels_participant = np.load(f"{PREPROCESSEDFOLDER}/{LABEL_FILE_PARTICIPANT}")
test_data = np.load(data_path)
labels_activity = np.load(labels_path_activity)
labels_participant = np.load(labels_path_participant)

x_data = test_data
y_label_activity = labels_activity
y_label_participant = labels_participant

(_, WIDTH, HEIGHT) = x_data.shape
CHANNELS = 1
print('WIDTH:', WIDTH)
print('HEIGHT:', HEIGHT)
print('CHANNELS:', CHANNELS)

# Reshape train datasets
x_data = x_data.reshape(
    x_data.shape[0],
    WIDTH,
    HEIGHT,
    CHANNELS
)

# Train a RandomForest model to get feature importances
rf_model = RandomForestClassifier()
# rf_model.fit(x_data.reshape(-1, WIDTH * HEIGHT * CHANNELS), y_label_activity)
rf_model.fit(x_data.reshape(-1, WIDTH * HEIGHT * CHANNELS), y_label_participant)
feature_importances = rf_model.feature_importances_

# Identify the most important features
print(f"Feature shape: {feature_importances.shape}")
k = 10000

# Get top k most important features
important_features = np.argsort(feature_importances)[-k:]
np.savetxt('important_features10000.txt', important_features, fmt='%.0f')

k = 1000
important_features = np.argsort(feature_importances)[-k:]
np.savetxt('important_features1000.txt', important_features, fmt='%.0f')
print("Saved Important Features")