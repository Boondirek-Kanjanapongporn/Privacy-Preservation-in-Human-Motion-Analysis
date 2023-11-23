import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle

PREPROCESSEDFOLDER = "D:/Individual Project/Preprocessed Participant Data"

# Alternative 1 --------------------------------------------------------------
TRAINDATASET = "trainDataset.npy"
TRAINPARTICIPANTLABEL = "trainlabel.npy"
VALIDATIONDATASET = "validationDataset.npy"
VALIDATIONLABEL = "validationlabel.npy"

# Load data and labels
train_data = np.load(f"{PREPROCESSEDFOLDER}/{TRAINDATASET}")
train_participant_labels = np.load(f"{PREPROCESSEDFOLDER}/{TRAINPARTICIPANTLABEL}")
validate_data = np.load(f"{PREPROCESSEDFOLDER}/{VALIDATIONDATASET}")
validate_participant_labels = np.load(f"{PREPROCESSEDFOLDER}/{VALIDATIONLABEL}")

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

# Extend Train label
label_data = np.copy(train_participant_labels)
train_participant_labels = np.concatenate((train_participant_labels, label_data))
train_participant_labels = np.concatenate((train_participant_labels, label_data))
train_participant_labels = np.concatenate((train_participant_labels, label_data))

# Shuffle data and labels
train_indices = np.arange(len(train_data))
np.random.shuffle(train_indices)
x_train = train_data[train_indices]
y_participant_train = train_participant_labels[train_indices]

validate_indices = np.arange(len(validate_data))
np.random.shuffle(validate_indices)
x_validate = validate_data[validate_indices]
y_participant_validate = validate_participant_labels[validate_indices]
# ----------------------------------------------------------------------------

# Print Data shape
print("x_train:", x_train.shape)
print("y_participant_train:", y_participant_train.shape)
print("x_validate:", x_validate.shape)
print("y_participant_validate:", y_participant_validate.shape)

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
# Define the input layer
input_layer = tf.keras.layers.Input(shape=(WIDTH, HEIGHT, CHANNELS))

# Shared convolutional layers
conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
pool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)
conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(pool1)
pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(pool2)
pool3 = tf.keras.layers.MaxPooling2D((2, 2))(conv3)
conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu')(pool3)
pool4 = tf.keras.layers.MaxPooling2D((2, 2))(conv4)
flatten = tf.keras.layers.Flatten()(pool4)

# Task-specific layers for activity recognition
activity_dense1 = tf.keras.layers.Dense(128, activation='relu')(flatten)
activity_dropout1 = tf.keras.layers.Dropout(0.4)(activity_dense1)
activity_dense2 = tf.keras.layers.Dense(64, activation='relu')(activity_dropout1)
activity_dropout2 = tf.keras.layers.Dropout(0.2)(activity_dense2)
activity_output = tf.keras.layers.Dense(6, activation='softmax', name='activity_output')(activity_dropout2)

# Task-specific layers for participant recognition
participant_dense1 = tf.keras.layers.Dense(128, activation='relu')(flatten)
participant_dropout1 = tf.keras.layers.Dropout(0.25)(participant_dense1)
participant_dense2 = tf.keras.layers.Dense(64, activation='relu')(participant_dropout1)
participant_dropout2 = tf.keras.layers.Dropout(0.5)(participant_dense2)
participant_output = tf.keras.layers.Dense(61, activation='softmax', name='participant_output')(participant_dropout2)

# Create a multi-task model
multi_task_model = tf.keras.models.Model(inputs=input_layer, outputs=[activity_output, participant_output])

# Compile the model with appropriate loss functions and metrics for each task
multi_task_model.compile(optimizer='adam',
                         loss={'activity_output': 'categorical_crossentropy', 'participant_output': 'categorical_crossentropy'},
                         loss_weights={'activity_output': 1.0, 'participant_output': 1.0},
                         metrics={'activity_output': 'accuracy', 'participant_output': 'accuracy'})

print(multi_task_model.summary())

print("Compile Model:")
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00015)
multi_task_model.compile(
    optimizer=optimizer,
    loss={'activity_output': 'categorical_crossentropy', 'participant_output': 'categorical_crossentropy'},
    loss_weights={'activity_output': 1.0, 'participant_output': 1.0},
    metrics={'activity_output': 'accuracy', 'participant_output': 'accuracy'}
)

print("Train Model:")
y_train = tf.keras.utils.to_categorical(y_train, num_classes=61)
y_validate = tf.keras.utils.to_categorical(y_validate, num_classes=61)

# Add early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Training parameters
train_epochs = 30
train_batch_size = 16

# Built-in Training loop -----------------------------------------------------
# Train model
training_history = multi_task_model.fit(
    x_train, 
    {'activity_output': y_activity_train, 'participant_output': y_participant_train}, 
    epochs=train_epochs,
    batch_size=train_batch_size, 
    validation_data=(x_validate, {'activity_output': y_activity_validate, 'participant_output': y_participant_validate}),
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