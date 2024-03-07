import tensorflow as tf
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import precision_score, recall_score, f1_score

PREPROCESSEDFOLDER = "D:/Individual Project/Preprocessed Participant Data"
# Alternative 1 --------------------------------------------------------------
# DATASET_FILE = "dataset1to6 Normalized R1.npy"
# LABEL_FILE = "dataset1to6 Label R1.npy"
# TRAINED_MODEL = "participant_recognition_cnn.h5"
# ----------------------------------------------------------------------------

# Alternative 2 --------------------------------------------------------------
DATASET_FILE = "testDataset30.npy"
LABEL_FILE = "testLabel30.npy"
TRAINED_MODEL = "experimentmodels/participant_recognition_cnn30.h5"
# ----------------------------------------------------------------------------

# Load data and labels
data = np.load(f"{PREPROCESSEDFOLDER}/{DATASET_FILE}")
labels = np.load(f"{PREPROCESSEDFOLDER}/{LABEL_FILE}")

# Shuffle data and labels
indices = np.arange(len(data))
np.random.shuffle(indices)
x_test = data[indices]
y_test = labels[indices]

# Print Data shape
print("x_test:", x_test.shape)
print("y_test:", y_test.shape)

# Load the model
loaded_model = tf.keras.models.load_model(TRAINED_MODEL)

# Predict the model with x_test
predictions_one_hot = loaded_model.predict([x_test])
print('predictions_one_hot:', predictions_one_hot.shape)

# Get most confident model prediction for each array
predictions = np.argmax(predictions_one_hot, axis=1)

# Plot the graph
# numbers_to_display = 61
numbers_to_display = 30
num_cells = math.ceil(math.sqrt(numbers_to_display))
plt.figure(figsize=(10,10))
for i in range(numbers_to_display):
    predicted_label = predictions[i]
    plt.subplot(num_cells, num_cells, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    color_map = 'Greens' if predicted_label == y_test[i] else 'Reds'
    img = plt.imshow(x_test[i], aspect='auto', cmap=color_map, extent=[0, 20, 13, -13])
    plt.xlabel(f"{predicted_label}" if predicted_label == y_test[i] else f"P: {predicted_label}, R: {y_test[i]}", fontsize=14, fontweight='bold')
    plt.ylim(-6, 6)
    clim = img.get_clim()
    plt.clim(clim[1]-0.6, clim[1])
plt.subplots_adjust(hspace=1, wspace=0.5)
plt.show(block=False)

# Find accuracy
accuracy = np.mean(y_test == predictions)
print(f"Accuracy (%) = {accuracy * 100:.2f}%")

# Calculate Precision, Recall, and F1-Score
precision = precision_score(y_test, predictions, average='macro')
recall = recall_score(y_test, predictions, average='macro')
f1 = f1_score(y_test, predictions, average='macro')

# Print the results
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

confusion_matrix = tf.math.confusion_matrix(predictions, y_test)
f, ax = plt.subplots(figsize=(14, 11))
sn.heatmap(
    confusion_matrix,
    annot=False,
    linewidths=.5,
    fmt="d",
    square=True,
    annot_kws={"size": 14},
    ax=ax
)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold', rotation=0)
plt.show()