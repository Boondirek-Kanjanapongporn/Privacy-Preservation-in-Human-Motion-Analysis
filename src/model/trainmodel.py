import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

PREPROCESSEDFOLDER = "D:/Individual Project/Preprocessed Data2"

# Alternative 1 --------------------------------------------------------------
DATASET_FILE = "dataset1to7 Normalized.npy"
LABEL_FILE = "dataset1to7 Label.npy"

# Load data and labels
data = np.load(f"{PREPROCESSEDFOLDER}/{DATASET_FILE}")
labels = np.load(f"{PREPROCESSEDFOLDER}/{LABEL_FILE}")

# Shuffle data and labels
indices = np.arange(len(data))
np.random.shuffle(indices)

shuffled_data = data[indices]
shuffled_labels = labels[indices]

# Define ratio for splitting up data and labels
train_ratio = 0.8

# Calculate the split point
split_point = int(len(data) * train_ratio)

# Split the data and labels
x_train, x_test = shuffled_data[:split_point], shuffled_data[split_point:]
y_train, y_test = shuffled_labels[:split_point], shuffled_labels[split_point:]

print("x_train:", x_train.shape)
print("x_test:", x_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
# ----------------------------------------------------------------------------

# Alternative 2 --------------------------------------------------------------
# X_TRAIN_FILE = "dataset1to6 Normalized.npy"
# Y_TRAIN_FILE = "dataset1to6 Label.npy"
# X_TEST_FILE = "7 March 2019 West Cumbria Dataset Normalized.npy"
# Y_TEST_FILE = "7 March 2019 West Cumbria Label.npy"

# # x_train
# x_train = np.load(f"{PREPROCESSEDFOLDER}/{X_TRAIN_FILE}")
# print(f"x train: {x_train.shape}")

# # y_train
# y_train = np.load(f"{PREPROCESSEDFOLDER}/{Y_TRAIN_FILE}")
# print(f"y train: {y_train.shape}")

# # x_test
# x_test = np.load(f"{PREPROCESSEDFOLDER}/{X_TEST_FILE}")
# print(f"x test: {x_test.shape}")

# # y_train
# y_test = np.load(f"{PREPROCESSEDFOLDER}/{Y_TEST_FILE}")
# print(f"y test: {y_test.shape}")
# ----------------------------------------------------------------------------

# Save image parameters to the constants that we will use later for data re-shaping and for model traning.
(_, WIDTH, HEIGHT) = x_train.shape
CHANNELS = 1
print('WIDTH:', WIDTH)
print('HEIGHT:', HEIGHT)
print('CHANNELS:', CHANNELS)

# Reshape train datasets
x_train = x_train.reshape(
    x_train.shape[0],
    WIDTH,
    HEIGHT,
    CHANNELS
)
x_test = x_test.reshape(
    x_test.shape[0],
    WIDTH,
    HEIGHT,
    CHANNELS
)
print('x_train:', x_train.shape)
print('x_test:', x_test.shape)

# Create Training Model
print("Building Model:")
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(WIDTH, HEIGHT, CHANNELS)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(6, activation='softmax')
])

print(model.summary())

print("Compile Model:")
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Train Model:")
y_train = tf.keras.utils.to_categorical(y_train, num_classes=6)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=6)

training_history = model.fit(
    x_train, 
    y_train, 
    epochs=10, 
    batch_size=32, 
    validation_data=(x_test, y_test)
)

plt.figure()
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.plot(training_history.history['loss'], label='training set')
plt.plot(training_history.history['val_loss'], label='test set')
plt.legend()
plt.show(block=False)

plt.figure()
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.plot(training_history.history['accuracy'], label='training set')
plt.plot(training_history.history['val_accuracy'], label='test set')
plt.legend()
plt.show()

# Training Loss
train_loss, train_accuracy = model.evaluate(x_train, y_train)
print('Training loss: ', train_loss)
print('Training accuracy: ', train_accuracy)

# Validation Loss
validation_loss, validation_accuracy = model.evaluate(x_test, y_test)
print('Validation loss: ', validation_loss)
print('Validation accuracy: ', validation_accuracy)

# Save Model
model_name = 'activity_recognition_cnn2.h5'
model.save(model_name, save_format='h5')