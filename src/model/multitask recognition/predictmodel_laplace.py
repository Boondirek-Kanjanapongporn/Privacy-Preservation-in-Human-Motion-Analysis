import tensorflow as tf
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import precision_score, recall_score, f1_score
from laplacefunctions import laplace_mechanism

PREPROCESSEDFOLDER = "D:/Individual Project/Preprocessed Multitask Data"

# Alternative 1 --------------------------------------------------------------
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

# Laplace epsilon value
epsilon = 0.14

# Predict the model with x_test
x_test_noisy = laplace_mechanism(x_test, epsilon)
predictions_one_hot_activity, predictions_one_hot_participant = loaded_model.predict([x_test_noisy])

# Apply Laplace mechanism
print('predictions_one_hot_activity:', predictions_one_hot_activity.shape)
print('predictions_one_hot_participant:', predictions_one_hot_participant.shape)

# Get most confident model prediction for each array
predictions_activity = np.argmax(predictions_one_hot_activity, axis=1)
predictions_participant = np.argmax(predictions_one_hot_participant, axis=1)

# Find accuracy
# Calculate accuracy for activity recognition
accuracy_activity = np.mean(predictions_activity == y_test_activity)
print(f"Activity Recognition Accuracy: {accuracy_activity * 100:.2f}%")

# Calculate accuracy for participant recognition
accuracy_participant = np.mean(predictions_participant == y_test_participant)
print(f"Participant Recognition Accuracy: {accuracy_participant * 100:.2f}%")

# Calculate Precision, Recall, and F1-Score
precision_activity = precision_score(y_test_activity, predictions_activity, average='macro')
recall_activity = recall_score(y_test_activity, predictions_activity, average='macro')
f1_activity = f1_score(y_test_activity, predictions_activity, average='macro')

# Calculate Precision, Recall, and F1-Score for participant recognition
precision_participant = precision_score(y_test_participant, predictions_participant, average='macro')
recall_participant = recall_score(y_test_participant, predictions_participant, average='macro')
f1_participant = f1_score(y_test_participant, predictions_participant, average='macro')

# Print the results for activity recognition
print("Activity Recognition Metrics:")
print(f"Precision: {precision_activity:.2f}")
print(f"Recall: {recall_activity:.2f}")
print(f"F1-Score: {f1_activity:.2f}")

# Print the results for participant recognition
print("Participant Recognition Metrics:")
print(f"Precision: {precision_participant:.2f}")
print(f"Recall: {recall_participant:.2f}")
print(f"F1-Score: {f1_participant:.2f}")

# Plot the graph activity
numbers_to_display = 30
num_cells = math.ceil(math.sqrt(numbers_to_display))
plt.figure(figsize=(10,10))
plt.title("Activity result")
plt.axis('off')
for i in range(numbers_to_display):
    predicted_label = predictions_activity[i]
    plt.subplot(num_cells, num_cells, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    color_map = 'Greens' if predicted_label == y_test_activity[i] else 'Reds'
    img = plt.imshow(x_test[i], aspect='auto', cmap=color_map, extent=[0, 20, 13, -13])
    plt.xlabel(f"{predicted_label}" if predicted_label == y_test_activity[i] else f"P: {predicted_label}, R: {y_test_activity[i]}", fontsize=14, fontweight='bold')
    plt.ylim(-6, 6)
    clim = img.get_clim()
    plt.clim(clim[1]-0.6, clim[1])
plt.subplots_adjust(hspace=1, wspace=0.5)
plt.show(block=False)

# Plot the graph participant
numbers_to_display = 30
num_cells = math.ceil(math.sqrt(numbers_to_display))
plt.figure(figsize=(10,10))
plt.title("Participant result")
plt.axis('off')
for i in range(numbers_to_display):
    predicted_label = predictions_participant[i]
    plt.subplot(num_cells, num_cells, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    color_map = 'Greens' if predicted_label == y_test_participant[i] else 'Reds'
    img = plt.imshow(x_test[i], aspect='auto', cmap=color_map, extent=[0, 20, 13, -13])
    plt.xlabel(f"{predicted_label}" if predicted_label == y_test_participant[i] else f"P: {predicted_label}, R: {y_test_participant[i]}", fontsize=14, fontweight='bold')
    plt.ylim(-6, 6)
    clim = img.get_clim()
    plt.clim(clim[1]-0.6, clim[1])
plt.subplots_adjust(hspace=1, wspace=0.5)
plt.show(block=True)

# Create confusion matrices
confusion_matrix_activity = tf.math.confusion_matrix(predictions_activity, y_test_activity)
confusion_matrix_participant = tf.math.confusion_matrix(predictions_participant, y_test_participant)

confusion_matrix_activity = tf.cast(confusion_matrix_activity, dtype=tf.float32)
confusion_matrix_activity_normalized = confusion_matrix_activity / (tf.reduce_sum(confusion_matrix_activity, axis=1, keepdims=True) + 1e-10)
activity_labels = ['Walk', 'Sit', 'Stand Up', 'Pick Up', 'Drink', 'Fall'] 
plt.figure(figsize=(14, 11))
ax = sn.heatmap(
    confusion_matrix_activity_normalized,
    annot=True,
    linewidths=.5,
    fmt=".2f",
    square=True,
    annot_kws={"size": 14},
    xticklabels=activity_labels,
    yticklabels=activity_labels,
)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.xlabel('Actual Label', fontsize=14, fontweight='bold')
plt.ylabel('Prediction Label', fontsize=14, fontweight='bold')
plt.show(block=False)

plt.figure(figsize=(14, 11))
sn.heatmap(
    confusion_matrix_participant,
    annot=False,
    linewidths=.5,
    fmt="d",
    square=True,
    annot_kws={"size": 14},
)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold', rotation=0)
plt.xlabel('Actual Label', fontsize=14, fontweight='bold')
plt.ylabel('Prediction Label', fontsize=14, fontweight='bold')
plt.show()