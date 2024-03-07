import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle

PREPROCESSEDFOLDER = "D:/Individual Project/Preprocessed Multitask Data"

# Alternative 1 --------------------------------------------------------------
TRAINDATASET = "trainDataset30.npy"
TRAINLABEL_ACTIVITY = "trainLabel_activity30.npy"
TRAINLABEL_PARTICIPANT = "trainLabel_participant30.npy"
VALIDATIONDATASET = "validationDataset30.npy"
VALIDATIONLABEL_ACTIVITY = "validationLabel_activity30.npy"
VALIDATIONLABEL_PARTICIPANT = "validationLabel_participant30.npy"

# Load data and labels
train_data = np.load(f"{PREPROCESSEDFOLDER}/{TRAINDATASET}")
train_labels_activity = np.load(f"{PREPROCESSEDFOLDER}/{TRAINLABEL_ACTIVITY}")
train_labels_participant = np.load(f"{PREPROCESSEDFOLDER}/{TRAINLABEL_PARTICIPANT}")
validate_data = np.load(f"{PREPROCESSEDFOLDER}/{VALIDATIONDATASET}")
validate_labels_activity = np.load(f"{PREPROCESSEDFOLDER}/{VALIDATIONLABEL_ACTIVITY}")
validate_labels_participant = np.load(f"{PREPROCESSEDFOLDER}/{VALIDATIONLABEL_PARTICIPANT}")

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

# Extend Train label Activity
label_data_activity = np.copy(train_labels_activity)
train_labels_activity = np.concatenate((train_labels_activity, label_data_activity))
train_labels_activity = np.concatenate((train_labels_activity, label_data_activity))
train_labels_activity = np.concatenate((train_labels_activity, label_data_activity))

# Extend Train label Participant
label_data_participant = np.copy(train_labels_participant)
train_labels_participant = np.concatenate((train_labels_participant, label_data_participant))
train_labels_participant = np.concatenate((train_labels_participant, label_data_participant))
train_labels_participant = np.concatenate((train_labels_participant, label_data_participant))

# Shuffle data and labels
train_indices = np.arange(len(train_data))
np.random.shuffle(train_indices)
x_train = train_data[train_indices]
y_train_activity = train_labels_activity[train_indices]
y_train_participant = train_labels_participant[train_indices]

validate_indices = np.arange(len(validate_data))
np.random.shuffle(validate_indices)
x_validate = validate_data[validate_indices]
y_validate_activity = validate_labels_activity[validate_indices]
y_validate_participant = validate_labels_participant[validate_indices]
# ----------------------------------------------------------------------------

# Print Data shape
print("x_train:", x_train.shape)
print("y_train_activity:", y_train_activity.shape)
print("y_train_participant:", y_train_participant.shape)
print("x_validate:", x_validate.shape)
print("y_validate_activity:", y_validate_activity.shape)
print("y_validate_participant:", y_validate_participant.shape)

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
conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')(input_layer)
pool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)
conv2 = tf.keras.layers.Conv2D(64, 3, activation='relu')(pool1)
pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
conv3 = tf.keras.layers.Conv2D(128, 3, activation='relu')(pool2)
pool3 = tf.keras.layers.MaxPooling2D((2, 2))(conv3)
conv4 = tf.keras.layers.Conv2D(256, 3, activation='relu')(pool3)
pool4 = tf.keras.layers.MaxPooling2D((2, 2))(conv4)
flatten = tf.keras.layers.Flatten()(pool4)

# Task-specific layers for activity recognition
activity_dense1 = tf.keras.layers.Dense(128, activation='relu')(flatten)
activity_dropout1 = tf.keras.layers.Dropout(0.4)(activity_dense1)
activity_dense2 = tf.keras.layers.Dense(64, activation='relu')(activity_dropout1)
# activity_dense2 = tf.keras.layers.Dense(64, activation='relu')(flatten)
activity_dropout2 = tf.keras.layers.Dropout(0.2)(activity_dense2)
activity_output = tf.keras.layers.Dense(6, activation='softmax', name='activity_output')(activity_dropout2)

# Task-specific layers for participant recognition
participant_dropout0 = tf.keras.layers.Dropout(0.25)(flatten)
participant_dense1 = tf.keras.layers.Dense(128, activation='relu')(participant_dropout0)
# participant_dense1 = tf.keras.layers.Dense(128, activation='relu')(flatten)
participant_dropout1 = tf.keras.layers.Dropout(0.3)(participant_dense1)
participant_dense2 = tf.keras.layers.Dense(64, activation='relu')(participant_dropout1)
participant_dropout2 = tf.keras.layers.Dropout(0.3)(participant_dense2)
participant_output = tf.keras.layers.Dense(30, activation='softmax', name='participant_output')(participant_dropout2)
# participant_output = tf.keras.layers.Dense(30, activation='softmax', name='participant_output')(participant_dense2)

# Create a multi-task model
model = tf.keras.models.Model(inputs=input_layer, outputs=[activity_output, participant_output])
print(model.summary())

# Compile the model with appropriate loss functions and metrics for each task
print("Compile Model:")
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00015)
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
model.compile(
    optimizer=optimizer,
    loss={'activity_output': 'categorical_crossentropy', 'participant_output': 'categorical_crossentropy'},
    loss_weights={'activity_output': 1.0, 'participant_output': 8.0},
    metrics={'activity_output': 'accuracy', 'participant_output': 'accuracy'}
)

print("Train Model:")
# Set categorical
y_train_activity = tf.keras.utils.to_categorical(y_train_activity, num_classes=6)
y_validate_activity = tf.keras.utils.to_categorical(y_validate_activity, num_classes=6)

y_train_participant = tf.keras.utils.to_categorical(y_train_participant, num_classes=30)
y_validate_participant = tf.keras.utils.to_categorical(y_validate_participant, num_classes=30)

# Add early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# Training parameters
train_epochs = 45
train_batch_size = 16
# train_epochs = 30
# train_batch_size = 16

# Custom Training loop -------------------------------------------------------
# def mixup_data(x, y_activity, y_participant, alpha=0.2):
#     # Applies the Mixup augmentation
#     if alpha > 0:
#         lam = np.random.beta(alpha, alpha)
#     else:
#         lam = 1

#     batch_size = x.shape[0]
#     index = np.random.permutation(batch_size)

#     mixed_x = lam * x + (1 - lam) * x[index, :]
#     mixed_y_activity = lam * y_activity + (1 - lam) * y_activity[index, :]
#     mixed_y_participant = lam * y_participant + (1 - lam) * y_participant[index, :]
#     return mixed_x, mixed_y_activity, mixed_y_participant

# # Create list for storing history footprint
# history_train_loss_activity = []
# history_train_accuracy_activity = []
# history_validation_loss_activity = []
# history_validation_accuracy_activity = []
# history_train_loss_participant = []
# history_train_accuracy_participant = []
# history_validation_loss_participant = []
# history_validation_accuracy_participant = []

# # with tf.device('/cpu:0'):
# # Custom training loop with Mixup
# for epoch in range(train_epochs):
#     print('Epoch', epoch + 1, '/', train_epochs)
#     x_train, y_train_activity, y_train_participant = shuffle(x_train, y_train_activity, y_train_participant)  # Shuffle the data in each epoch
#     for i in range(0, x_train.shape[0], train_batch_size):
#         x_batch = x_train[i:i + train_batch_size]
#         y_batch_activity = y_train_activity[i:i + train_batch_size]
#         y_batch_participant = y_train_participant[i:i + train_batch_size]

#         # Apply Mixup
#         x_batch, y_batch_activity, y_batch_participant = mixup_data(x_batch, y_batch_activity, y_batch_participant, alpha=0.2)

#         # Train on batch
#         model.train_on_batch(x_batch, {'activity_output': y_batch_activity, 'participant_output': y_batch_participant})

#     # Validate after each epoch
#     _, activity_loss_train, participant_loss_train, activity_accuracy_train, participant_accuracy_train = model.evaluate(x_train, {'activity_output': y_train_activity, 'participant_output': y_train_participant})
#     print('Training loss activity:', activity_loss_train)
#     print('Training accuracy activity:', activity_accuracy_train)
#     print('Training loss participant:', participant_loss_train)
#     print('Training accuracy participant:', participant_accuracy_train)

#     _, activity_loss_validation, participant_loss_validation, activity_accuracy_validation, participant_accuracy_validation = model.evaluate(x_validate, {'activity_output': y_validate_activity, 'participant_output': y_validate_participant})
#     print('Validation loss activity:', activity_loss_validation)
#     print('Validation accuracy activity:', activity_accuracy_validation)
#     print('Validation loss participant:', participant_loss_validation)
#     print('Validation accuracy participant:', participant_accuracy_validation)

#     # Store into history
#     history_train_loss_activity.append(activity_loss_train)
#     history_train_accuracy_activity.append(activity_accuracy_train)
#     history_validation_loss_activity.append(activity_loss_validation)
#     history_validation_accuracy_activity.append(activity_accuracy_validation)

#     history_train_loss_participant.append(participant_loss_train)
#     history_train_accuracy_participant.append(participant_accuracy_train)
#     history_validation_loss_participant.append(participant_loss_validation)
#     history_validation_accuracy_participant.append(participant_accuracy_validation)

#     # Early stopping check
#     if early_stopping.model is not None and early_stopping.stopped_epoch > 0:
#         break

# plt.figure()
# plt.xlabel('Epoch Number')
# plt.ylabel('Loss')
# plt.plot(history_train_loss_activity, label='training set')
# plt.plot(history_validation_loss_activity, label='validation set')
# plt.legend()
# plt.show(block=False)

# plt.figure()
# plt.xlabel('Epoch Number')
# plt.ylabel('Accuracy')
# plt.plot(history_train_accuracy_activity, label='training set')
# plt.plot(history_validation_accuracy_activity, label='validation set')
# plt.legend()
# plt.show()

# plt.figure()
# plt.xlabel('Epoch Number')
# plt.ylabel('Loss')
# plt.plot(history_train_loss_participant, label='training set')
# plt.plot(history_validation_loss_participant, label='validation set')
# plt.legend()
# plt.show(block=False)

# plt.figure()
# plt.xlabel('Epoch Number')
# plt.ylabel('Accuracy')
# plt.plot(history_train_accuracy_participant, label='training set')
# plt.plot(history_validation_accuracy_participant, label='validation set')
# plt.legend()
# plt.show()
# ----------------------------------------------------------------------------

# Built-in Training loop -----------------------------------------------------
# Train model
training_history = model.fit(
    x_train, 
    {'activity_output': y_train_activity, 'participant_output': y_train_participant}, 
    epochs=train_epochs,
    batch_size=train_batch_size, 
    validation_data=(x_validate, {'activity_output': y_validate_activity, 'participant_output': y_validate_participant}),
    callbacks=[early_stopping]
)

# Plot loss for activity recognition
fig, ax1 = plt.subplots(figsize=(14, 7))  # Adjusted subplot size
ax1.set_xlabel('Epoch Number', fontsize=16, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=16, fontweight='bold', color='tab:red')
ax1.plot(training_history.history['activity_output_loss'], label='training loss', color='tab:red', linestyle='--')
ax1.plot(training_history.history['val_activity_output_loss'], label='validation loss', color='tab:pink', linestyle='--')
ax1.tick_params(axis='y', labelcolor='tab:red', labelsize=16)

# Instantiate a second y-axis sharing the same x-axis
ax2 = ax1.twinx()

# Plot accuracy for activity recognition
ax2.set_ylabel('Accuracy', fontsize=16, fontweight='bold', color='tab:blue')
ax2.plot(training_history.history['activity_output_accuracy'], label='training accuracy', color='tab:blue')
ax2.plot(training_history.history['val_activity_output_accuracy'], label='validation accuracy', color='tab:cyan')
ax2.tick_params(axis='y', labelcolor='tab:blue', labelsize=16)

# Combine legends from both subplots
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax2.legend(lines_2 + lines_1, labels_2 + labels_1, loc='upper center', fontsize=14)

plt.title('Activity Training Result', fontsize=16, fontweight='bold')
plt.xlabel('Epoch Number', fontsize=16, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show(block=False)

# Plot loss for participant recognition
fig, ax1 = plt.subplots(figsize=(14, 7))
ax1.set_xlabel('Epoch Number', fontsize=16, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=16, fontweight='bold', color='tab:red')
ax1.plot(training_history.history['participant_output_loss'], label='training loss', color='tab:red', linestyle='--')
ax1.plot(training_history.history['val_participant_output_loss'], label='validation loss', color='tab:pink', linestyle='--')
ax1.tick_params(axis='y', labelcolor='tab:red', labelsize=16)

# Instantiate a second y-axis sharing the same x-axis
ax2 = ax1.twinx()

# Plot accuracy for participant recognition
ax2.set_xlabel('Epoch Number', fontsize=16, fontweight='bold')
ax2.set_ylabel('Accuracy', fontsize=16, fontweight='bold', color='tab:blue')
ax2.plot(training_history.history['participant_output_accuracy'], label='training accuracy', color='tab:blue')
ax2.plot(training_history.history['val_participant_output_accuracy'], label='validation accuracy', color='tab:cyan')
ax2.tick_params(axis='y', labelcolor='tab:blue', labelsize=16)

# Combine legends from both subplots
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax2.legend(lines_2 + lines_1, labels_2 + labels_1, loc='center left', fontsize=14)

plt.title('Participant Training Result', fontsize=16, fontweight='bold')
plt.xlabel('Epoch Number', fontsize=16, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show(block=True)
# ----------------------------------------------------------------------------
# Save Model
model_name = 'experimentmodels/multitask_recognition_cnn.h5'
model.save(model_name, save_format='h5')

# Evaluate results
_, activity_loss_train, participant_loss_train, activity_accuracy_train, participant_accuracy_train = model.evaluate(x_train, {'activity_output': y_train_activity, 'participant_output': y_train_participant})
_, activity_loss_validation, participant_loss_validation, activity_accuracy_validation, participant_accuracy_validation = model.evaluate(x_validate, {'activity_output': y_validate_activity, 'participant_output': y_validate_participant})

# Evaluate for Train dataset
print('Activity Recognition:')
print('Activity Training Loss: ', activity_loss_train)
print('Activity Training Accuracy: ', activity_accuracy_train)
print('Activity Validation Loss: ', activity_loss_validation)
print('Activity Validation Accuracy', activity_accuracy_validation)

# Evaluate for Participant
print('\nParticipant Recognition:')
print('Participant Training Loss: ', participant_loss_train)
print('Participant Training Accuracy: ', participant_accuracy_train)
print('Participant Validation Loss: ', participant_loss_validation)
print('Participant Validation Accuracy', participant_accuracy_validation)