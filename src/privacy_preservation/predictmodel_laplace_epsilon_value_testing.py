import tensorflow as tf
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import precision_score, recall_score, f1_score
from pathlib import Path

# PREPROCESSEDFOLDER = "D:/Individual Project/Preprocessed Multitask Data"
PREPROCESSEDFOLDER = "../../data/processed"

# Alternative 2 --------------------------------------------------------------
TESTDATASET_FILE = "testDataset30.npy"
LABEL_FILE_ACTIVITY = "testLabel_activity30.npy"
LABEL_FILE_PARTICIPANT = "testLabel_participant30.npy"
TRAINED_MODEL = "experimentmodels/multitask_recognition_cnn5_97_70.h5"

script_location = Path(__file__).resolve().parent
data_folder = script_location / PREPROCESSEDFOLDER
data_path = data_folder / TESTDATASET_FILE
labels_path_activity = data_folder / LABEL_FILE_ACTIVITY 
labels_path_participant = data_folder / LABEL_FILE_PARTICIPANT
# ----------------------------------------------------------------------------

# Load data and labels
# test_data = np.load(f"{PREPROCESSEDFOLDER}/{TESTDATASET_FILE}")
# labels_activity = np.load(f"{PREPROCESSEDFOLDER}/{LABEL_FILE_ACTIVITY}")
# labels_participant = np.load(f"{PREPROCESSEDFOLDER}/{LABEL_FILE_PARTICIPANT}")
test_data = np.load(data_path)
labels_activity = np.load(labels_path_activity)
labels_participant = np.load(labels_path_participant)

# Shuffle data and labels
indices = np.arange(len(test_data))
np.random.shuffle(indices)
x_test = test_data[indices]
y_test_activity = labels_activity[indices]
y_test_participant = labels_participant[indices]

# Print Data shape
print("x_test:", x_test.shape)
print("y_test_activity:", y_test_activity.shape)
print("y_test_participant:", y_test_participant.shape)

# Load the model
loaded_model = tf.keras.models.load_model(TRAINED_MODEL)

# Laplace Mechanism
def laplace_mechanism(data, epsilon):
    # Generate and add Laplace noise
    noise = np.random.laplace(0, epsilon, data.shape)
    noisy_data = data + noise
    return noisy_data

# Laplace epsilon value
epsilon = 0
epsilon_accuracies_activity = []
epsilon_f1_scores_activity = []
epsilon_accuracies_participant = []
epsilon_f1_scores_participant = []
for i in range(41):
    accuracy_list_activity = []
    f1_score_list_activity = []
    accuracy_list_participant = []
    f1_score_list_participant = []
    print(f"Epsilon Value: {epsilon}")

    for j in range(10):
        # Predict the model with x_test
        x_test_added_noised = laplace_mechanism(x_test, epsilon)

        predictions_one_hot_activity, predictions_one_hot_participant = loaded_model.predict([x_test_added_noised], verbose=0)

        # Get most confident model prediction for each array
        predictions_activity = np.argmax(predictions_one_hot_activity, axis=1)
        predictions_participant = np.argmax(predictions_one_hot_participant, axis=1)

        # Find accuracy
        # # Calculate accuracy for activity recognition
        accuracy_activity = np.mean(predictions_activity == y_test_activity)
        accuracy_list_activity.append(accuracy_activity * 100)

        # Calculate accuracy for participant recognition
        accuracy_participant = np.mean(predictions_participant == y_test_participant)
        accuracy_list_participant.append(accuracy_participant * 100)

        # # Calculate F1-Score for activity classification
        f1_activity = f1_score(y_test_activity, predictions_activity, average='macro')
        f1_score_list_activity.append(f1_activity)

        # Calculate F1-Score for participant recognition
        f1_participant = f1_score(y_test_participant, predictions_participant, average='macro')
        f1_score_list_participant.append(f1_participant)

    # # Print the results for activity recognition
    # print("Activity Recognition Metrics:")
    # print(f"Precision: {precision_activity:.2f}")
    # print(f"Recall: {recall_activity:.2f}")
    # print(f"F1-Score: {f1_activity:.2f}")

    accuracy_array_activity = np.array(accuracy_list_activity)
    f1_score_array_activity = np.array(f1_score_list_activity)
    accuracy_array_participant = np.array(accuracy_list_participant)
    f1_score_array_participant = np.array(f1_score_list_participant)

    # Print the results for activity classification
    print("Activity classification Metrics Average:")
    print(f"Accuracy: {np.average(accuracy_array_activity):.2f}")
    print(f"F1-Score: {np.average(f1_score_array_activity):.2f}")

    # Print the results for participant recognition
    print("Participant Recognition Metrics Average:")
    print(f"Accuracy: {np.average(accuracy_array_participant):.2f}")
    print(f"F1-Score: {np.average(f1_score_array_participant):.2f}")

    epsilon_accuracies_activity.append(np.average(accuracy_array_activity))
    epsilon_f1_scores_activity.append(np.average(f1_score_array_activity))
    epsilon_accuracies_participant.append(np.average(accuracy_array_participant))
    epsilon_f1_scores_participant.append(np.average(f1_score_array_participant))
    print("\n")

    epsilon = np.round(epsilon + 0.01, 2)

print("Epsilon Accuracies Activity:")
print(epsilon_accuracies_activity)

print("Epsilon F1 score Activity:")
print(epsilon_f1_scores_activity)

print("\n")

print("Epsilon Accuracies Participant:")
print(epsilon_accuracies_participant)

print("Epsilon F1 score Participant:")
print(epsilon_f1_scores_participant)