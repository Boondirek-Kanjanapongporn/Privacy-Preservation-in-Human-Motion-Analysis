import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import optuna

PREPROCESSEDFOLDER = "D:/Individual Project/Preprocessed Participant Data"

# Load Data  --------------------------------------------------------------
TRAINDATASET = "trainDataset_walk.npy"
TRAINLABEL = "datasetLabel.npy"
VALIDATIONDATASET = "validationDataset_walk.npy"
VALIDATIONLABEL = "datasetLabel.npy"

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
print("x_validate:", x_validate.shape)
print("y_train:", y_train.shape)
print("y_validate:", y_validate.shape)

# Process/Modify Loaded Data -------------------------------------------------
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

# Process labels to have correct shape (x, 6)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=61)
y_validate = tf.keras.utils.to_categorical(y_validate, num_classes=61)
# ----------------------------------------------------------------------------

# Hyperparameters Optimization -----------------------------------------------
def create_model(trial):
    global WIDTH, HEIGHT, CHANNELS
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(trial.suggest_int("conv1_filters", 32, 256), trial.suggest_int("kernel1", 2, 5), activation='relu', input_shape=(WIDTH, HEIGHT, CHANNELS)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(trial.suggest_int("conv2_filters", 32, 256), trial.suggest_int("kernel2", 2, 5), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(trial.suggest_int("conv3_filters", 32, 256), trial.suggest_int("kernel3", 2, 5), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(trial.suggest_int("dense_units1", 64, 256), activation='relu'))
    model.add(tf.keras.layers.Dropout(trial.suggest_float("dropout1", 0.1, 0.5)))
    model.add(tf.keras.layers.Dense(trial.suggest_int("dense_units2", 64, 128), activation='relu'))
    model.add(tf.keras.layers.Dropout(trial.suggest_float("dropout2", 0.1, 0.5)))
    model.add(tf.keras.layers.Dense(61, activation='softmax'))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=trial.suggest_float("learning_rate", 0.0005, 0.002))
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Add early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

def objective(trial):
    global x_train, y_train, x_validate, y_validate

    # Create a model with trial-specific hyperparameters
    model = create_model(trial)
    
    # Train the model
    training_history = model.fit(
        x_train, 
        y_train, 
        epochs=30,
        batch_size=16,
        validation_data=(x_validate, y_validate),
        verbose=0,  # Set to 0 to avoid printing progress for each trial
        callbacks=[early_stopping]
    )

    # Evaluate the model on the validation set
    validation_loss, validation_accuracy = model.evaluate(x_validate, y_validate)

    return validation_loss
# ----------------------------------------------------------------------------

# Define the search space for hyperparameters
study = optuna.create_study(direction="minimize")
with tf.device('/cpu:0'):
    study.optimize(objective, n_trials=100)

# Get the best hyperparameters
best_params = study.best_params
print("Best Hyperparameters:", best_params)

# Rebuild and train the model with the best hyperparameters
best_model = create_model(best_params)
training_history = best_model.fit(
    x_train, 
    y_train, 
    epochs=30,
    batch_size=16, 
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

# Evaluate the best model
best_validation_loss, best_validation_accuracy = best_model.evaluate(x_validate, y_validate)
print('Best Validation loss: ', best_validation_loss)
print('Best Validation accuracy: ', best_validation_accuracy)

# Save the best model
model_name = 'best_participant_recognition_cnn.h5'
best_model.save(model_name, save_format='h5')
