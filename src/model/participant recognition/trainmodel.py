import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle

def mixup_data(x, y, alpha=0.2):
    # Applies the Mixup augmentation
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.shape[0]
    index = np.random.permutation(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    return mixed_x, mixed_y

PREPROCESSEDFOLDER = "D:/Individual Project/Preprocessed Participant Data"

# Alternative 1 --------------------------------------------------------------
# DATASET_FILE = "dataset1to6 Normalized R2&R3 + DA.npy"
# LABEL_FILE = "dataset1to6 Label R2&R3 + DA.npy"

# # Load data and labels
# data = np.load(f"{PREPROCESSEDFOLDER}/{DATASET_FILE}")
# labels = np.load(f"{PREPROCESSEDFOLDER}/{LABEL_FILE}")

# # Shuffle data and labels
# indices = np.arange(len(data))
# np.random.shuffle(indices)

# shuffled_data = data[indices]
# shuffled_labels = labels[indices]

# # Define ratio for splitting up data and labels into training and validation dataset
# train_ratio = 0.75

# # Calculate the split point
# split_point = int(len(data) * train_ratio)

# # Split the data and labels into training and validate set
# x_train, x_validate = shuffled_data[:split_point], shuffled_data[split_point:]
# y_train, y_validate = shuffled_labels[:split_point], shuffled_labels[split_point:]
# ----------------------------------------------------------------------------

# Alternative 2 --------------------------------------------------------------
# TRAINDATASET = "trainDataset.npy"
# TRAINLABEL = "trainLabel.npy"
# VALIDATIONDATASET = "validationDataset.npy"
# VALIDATIONLABEL = "validationLabel.npy"
TRAINDATASET = "trainDataset1.npy"
TRAINLABEL = "trainLabel1.npy"
VALIDATIONDATASET = "validationDataset1.npy"
VALIDATIONLABEL = "validationLabel1.npy"

# Load data and labels
train_data = np.load(f"{PREPROCESSEDFOLDER}/{TRAINDATASET}")
train_labels = np.load(f"{PREPROCESSEDFOLDER}/{TRAINLABEL}")
validate_data = np.load(f"{PREPROCESSEDFOLDER}/{VALIDATIONDATASET}")
validate_labels = np.load(f"{PREPROCESSEDFOLDER}/{VALIDATIONLABEL}")

# Data Augment on Train data
train_data_fliplr = []
train_data_flipud = []
train_data_fliplr_flipud = []
for d in train_data:
    train_data_fliplr.append(np.fliplr(d))
    train_data_flipud.append(np.flipud(d))
    train_data_fliplr_flipud.append(np.fliplr(np.flipud(d)))
train_data_fliplr = np.array(train_data_fliplr)
train_data_flipud = np.array(train_data_flipud)
train_data_fliplr_flipud = np.array(train_data_fliplr_flipud)
train_data = np.concatenate((train_data, train_data_fliplr))
train_data = np.concatenate((train_data, train_data_flipud))
train_data = np.concatenate((train_data, train_data_fliplr_flipud))
# print(f"Train Dataset (Data Augmented): {train_data.shape}")

# Extend Train label
label_data = np.copy(train_labels)
train_labels = np.concatenate((train_labels, label_data))
train_labels = np.concatenate((train_labels, label_data))
train_labels = np.concatenate((train_labels, label_data))
# print(f"Train Label: {train_labels.shape}")

# Shuffle data and labels
train_indices = np.arange(len(train_data))
np.random.shuffle(train_indices)
x_train = train_data[train_indices]
y_train = train_labels[train_indices]

validate_indices = np.arange(len(validate_data))
np.random.shuffle(validate_indices)
x_validate = validate_data[validate_indices]
y_validate = validate_labels[validate_indices]
# ----------------------------------------------------------------------------

# Print Data shape
print("x_train:", x_train.shape)
print("y_train:", y_train.shape)
print("x_validate:", x_validate.shape)
print("y_validate:", y_validate.shape)

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
x_validate = x_validate.reshape(
    x_validate.shape[0],
    WIDTH,
    HEIGHT,
    CHANNELS
)
print('x_train:', x_train.shape)
print('x_validate:', x_validate.shape)

# Create Training Model
print("Building Model:")
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (5, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (5, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(256, (5, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    # tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(61, activation='softmax')
])

print(model.summary())

print("Compile Model:")
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Train Model:")

y_train = tf.keras.utils.to_categorical(y_train, num_classes=61)
y_validate = tf.keras.utils.to_categorical(y_validate, num_classes=61)

# Add early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Training parameters
train_epochs = 30
train_batch_size = 16

# Custom Training loop -------------------------------------------------------
# # Create list for storing history footprint
# history_train_loss = []
# history_train_accuracy = []
# history_validation_loss = []
# history_validation_accuracy = []

# with tf.device('/cpu:0'):
#     # Custom training loop with Mixup
#     for epoch in range(train_epochs):
#         print('Epoch', epoch + 1, '/', train_epochs)
#         x_train, y_train = shuffle(x_train, y_train)  # Shuffle the data in each epoch
#         for i in range(0, x_train.shape[0], train_batch_size):
#             x_batch = x_train[i:i + train_batch_size]
#             y_batch = y_train[i:i + train_batch_size]

#             # Apply Mixup
#             x_batch, y_batch = mixup_data(x_batch, y_batch, alpha=0.2)

#             # Train on batch
#             model.train_on_batch(x_batch, y_batch)

#         # Validate after each epoch
#         train_loss, train_acc = model.evaluate(x_train, y_train)
#         print('Training loss:', train_loss)
#         print('Training accuracy:', train_acc)

#         val_loss, val_acc = model.evaluate(x_validate, y_validate)
#         print('Validation loss:', val_loss)
#         print('Validation accuracy:', val_acc)

#         # Store into history
#         history_train_loss.append(train_loss)
#         history_train_accuracy.append(train_acc)
#         history_validation_loss.append(val_loss)
#         history_validation_accuracy.append(val_acc)

#         # Early stopping check
#         if early_stopping.model is not None and early_stopping.stopped_epoch > 0:
#             break

# plt.figure()
# plt.xlabel('Epoch Number')
# plt.ylabel('Loss')
# plt.plot(history_train_loss, label='training set')
# plt.plot(history_validation_loss, label='validation set')
# plt.legend()
# plt.show(block=False)

# plt.figure()
# plt.xlabel('Epoch Number')
# plt.ylabel('Accuracy')
# plt.plot(history_train_accuracy, label='training set')
# plt.plot(history_validation_accuracy, label='validation set')
# plt.legend()
# plt.show()
# ----------------------------------------------------------------------------

# Built-in Training loop -----------------------------------------------------
# Train model
training_history = model.fit(
    x_train, 
    y_train, 
    epochs=train_epochs,
    batch_size=train_batch_size, 
    validation_data=(x_validate, y_validate),
    callbacks=[early_stopping]
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
# ----------------------------------------------------------------------------

# Training Loss
train_loss, train_accuracy = model.evaluate(x_train, y_train)
print('Training loss: ', train_loss)
print('Training accuracy: ', train_accuracy)

# Validation Loss
validation_loss, validation_accuracy = model.evaluate(x_validate, y_validate)
print('Validation loss: ', validation_loss)
print('Validation accuracy: ', validation_accuracy)

# Save Model
model_name = 'participant_recognition_cnn.h5'
model.save(model_name, save_format='h5')