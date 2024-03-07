import tensorflow as tf
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import precision_score, recall_score, f1_score

PREPROCESSEDFOLDER = "D:/Individual Project/Preprocessed Multitask Data"

# Alternative 2 --------------------------------------------------------------
TESTDATASET_FILE = "testDataset30.npy"
LABEL_FILE_ACTIVITY = "testLabel_activity30.npy"
LABEL_FILE_PARTICIPANT = "testLabel_participant30.npy"
TRAINED_MODEL = "experimentmodels/multitask_recognition_cnn5_97_70.h5"
# ----------------------------------------------------------------------------

# Load data and labels
test_data = np.load(f"{PREPROCESSEDFOLDER}/{TESTDATASET_FILE}")
labels_activity = np.load(f"{PREPROCESSEDFOLDER}/{LABEL_FILE_ACTIVITY}")
labels_participant = np.load(f"{PREPROCESSEDFOLDER}/{LABEL_FILE_PARTICIPANT}")

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
epsilon_participant = 0
epsilon_accuracies = []
epsilon_f1_scores = []
for i in range(41):
    accuracy_list = []
    f1_score_list = []
    print(f"Epsilon Value: {epsilon_participant}")

    for j in range(10):
        # Predict the model with x_test
        predictions_one_hot_activity, predictions_one_hot_participant = loaded_model.predict([x_test], verbose=0)
        predictions_one_hot_participant = laplace_mechanism(predictions_one_hot_participant, epsilon_participant)

        # Get most confident model prediction for each array
        predictions_activity = np.argmax(predictions_one_hot_activity, axis=1)
        predictions_participant = np.argmax(predictions_one_hot_participant, axis=1)

        # Find accuracy
        # # Calculate accuracy for activity recognition
        # accuracy_activity = np.mean(predictions_activity == y_test_activity)
        # print(f"Activity Recognition Accuracy: {accuracy_activity * 100:.2f}%")

        # Calculate accuracy for participant recognition
        accuracy_participant = np.mean(predictions_participant == y_test_participant)
        accuracy_list.append(accuracy_participant * 100)

        # # Calculate Precision, Recall, and F1-Score
        # precision_activity = precision_score(y_test_activity, predictions_activity, average='macro')
        # recall_activity = recall_score(y_test_activity, predictions_activity, average='macro')
        # f1_activity = f1_score(y_test_activity, predictions_activity, average='macro')

        # Calculate F1-Score for participant recognition
        f1_participant = f1_score(y_test_participant, predictions_participant, average='macro')
        f1_score_list.append(f1_participant)

    # # Print the results for activity recognition
    # print("Activity Recognition Metrics:")
    # print(f"Precision: {precision_activity:.2f}")
    # print(f"Recall: {recall_activity:.2f}")
    # print(f"F1-Score: {f1_activity:.2f}")

    accuracy_array = np.array(accuracy_list)
    f1_score_array = np.array(f1_score_list)

    # Print the results for participant recognition
    print("Participant Recognition Metrics Average:")
    print(f"Accuracy: {np.average(accuracy_array):.2f}")
    print(f"F1-Score: {np.average(f1_score_array):.2f}")

    epsilon_accuracies.append(np.average(accuracy_array))
    epsilon_f1_scores.append(np.average(f1_score_array))
    print("\n")

    epsilon_participant = np.round(epsilon_participant + 0.03, 2)

print("Epsilon Accuracies:")
print(epsilon_accuracies)

print("Epsilon F1 score:")
print(epsilon_f1_scores)