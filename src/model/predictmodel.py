import tensorflow as tf
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sn

PREPROCESSEDFOLDER = "D:/Individual Project/Preprocessed Activity Data"
DATASET_FILE = "dataset1to7 Normalized.npy"
LABEL_FILE = "dataset1to7 Label.npy"
TRAINED_MODEL = "activity_recognition_cnn.h5"

# Load data and labels
data = np.load(f"{PREPROCESSEDFOLDER}/{DATASET_FILE}")
labels = np.load(f"{PREPROCESSEDFOLDER}/{LABEL_FILE}")

# Split out data for training + validating and testing data
START, END = 50, 50+260
data, labels = data[START:END], labels[START:END]

# Shuffle data and labels
indices = np.arange(len(data))
np.random.shuffle(indices)

x_test = data[indices]
y_test = labels[indices]

print("x_test:", x_test.shape)
print("y_test:", y_test.shape)

# Load the model
loaded_model = tf.keras.models.load_model(TRAINED_MODEL)

# Predict the model with x_test
predictions_one_hot = loaded_model.predict([x_test])
print('predictions_one_hot:', predictions_one_hot.shape)

# Get most confident model prediction for each array
predictions = np.argmax(predictions_one_hot, axis=1)
pd.DataFrame(predictions)

# Plot the graph
numbers_to_display = 200
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
    plt.xlabel(f"{predicted_label}" if predicted_label == y_test[i] else f"P: {predicted_label}, R: {y_test[i]}")
    plt.ylim(-6, 6)
    clim = img.get_clim()
    plt.clim(clim[1]-0.6, clim[1])

plt.subplots_adjust(hspace=1, wspace=0.5)
plt.show(block=False)

confusion_matrix = tf.math.confusion_matrix(predictions, y_test)
f, ax = plt.subplots(figsize=(9, 7))
sn.heatmap(
    confusion_matrix,
    annot=True,
    linewidths=.5,
    fmt="d",
    square=True,
    ax=ax
)
plt.show()