import tensorflow as tf
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import precision_score, recall_score, f1_score
from laplacefunctions import *

PREPROCESSEDFOLDER = "D:/Individual Project/Preprocessed Multitask Data"

# Alternative 1 --------------------------------------------------------------
TESTDATASET_FILE = "testDataset30.npy"
LABEL_FILE_ACTIVITY = "testLabel_activity30.npy"
LABEL_FILE_PARTICIPANT = "testLabel_participant30.npy"
TRAINED_MODEL = "experimentmodels/multitask_recognition_cnn_final.h5"
# ----------------------------------------------------------------------------
with tf.device('/cpu:0'):
    k = 0
    for _ in range(1):
        k += 1000
        print(f"k value: {k}")

        # Epsilon value setting
        epsilon = 0
        weighted_epsilon = 0

        activity_f1_scores = []
        activity_accuracies = []
        participant_f1_scores = []
        participant_accuracies = []
        for _ in range(45):
            print(f"Weight epsilon: {weighted_epsilon}")
            # Load data and labels
            test_data = np.load(f"{PREPROCESSEDFOLDER}/{TESTDATASET_FILE}")
            labels_activity = np.load(f"{PREPROCESSEDFOLDER}/{LABEL_FILE_ACTIVITY}")
            labels_participant = np.load(f"{PREPROCESSEDFOLDER}/{LABEL_FILE_PARTICIPANT}")

            (PARTICIPANTS, WIDTH, HEIGHT) = test_data.shape
            CHANNELS = 1

            test_data = weighted_laplace_mechanism(test_data.reshape(-1, WIDTH * HEIGHT * CHANNELS), epsilon, weighted_epsilon, important_features_participant4[-k:])
            test_data = test_data.reshape(PARTICIPANTS, WIDTH, HEIGHT)

            # Shuffle data and labels
            indices = np.arange(len(test_data))
            np.random.shuffle(indices)
            x_test = test_data[indices]
            y_test_activity = labels_activity[indices]
            y_test_participant = labels_participant[indices]

            # Load the model
            loaded_model = tf.keras.models.load_model(TRAINED_MODEL)

            # Predict the model with x_test
            predictions_one_hot_activity, predictions_one_hot_participant = loaded_model.predict([x_test], verbose=0)

            # Get most confident model prediction for each array
            predictions_activity = np.argmax(predictions_one_hot_activity, axis=1)
            predictions_participant = np.argmax(predictions_one_hot_participant, axis=1)

            # Find accuracy
            # Calculate accuracy for activity recognition
            accuracy_activity = np.mean(predictions_activity == y_test_activity)
            # print(f"Activity Recognition Accuracy: {accuracy_activity * 100:.2f}%")
            activity_accuracies.append(np.round(accuracy_activity * 100, 2))

            # Calculate accuracy for participant recognition
            accuracy_participant = np.mean(predictions_participant == y_test_participant)
            # print(f"Participant Recognition Accuracy: {accuracy_participant * 100:.2f}%\n")
            participant_accuracies.append(np.round(accuracy_participant * 100, 2))

            # Calculate Precision, Recall, and F1-Score
            precision_activity = precision_score(y_test_activity, predictions_activity, average='macro')
            recall_activity = recall_score(y_test_activity, predictions_activity, average='macro')
            f1_activity = f1_score(y_test_activity, predictions_activity, average='macro')
            activity_f1_scores.append(np.round(f1_activity, 3))

            # Calculate Precision, Recall, and F1-Score for participant recognition
            precision_participant = precision_score(y_test_participant, predictions_participant, average='macro')
            recall_participant = recall_score(y_test_participant, predictions_participant, average='macro')
            f1_participant = f1_score(y_test_participant, predictions_participant, average='macro')
            participant_f1_scores.append(np.round(f1_participant, 3))

            # Update weighted epsilon
            weighted_epsilon = np.round(weighted_epsilon + 0.03, 2)

        print(f"Activity F1-scores: {activity_f1_scores}\n")
        print(f"Participant F1-scores: {participant_f1_scores}\n")
        # print(f"Activity Accuracies: {activity_accuracies}\n")
        # print(f"Participant Accuracies: {participant_accuracies}")
    