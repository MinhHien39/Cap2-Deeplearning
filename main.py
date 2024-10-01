import tensorflow as tf
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, confusion_matrix, recall_score
from tensorflow.keras import layers, models, Input
from keras.src.legacy.preprocessing.image import ImageDataGenerator

# Data Preparation
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Define source and destination directories
train_source_dir = "./Model/Train"
test_source_dir = "./Model/Test"
train_sep_dir = "train_sep"
test_sep_dir = "test_sep"

# Define subfolders to be copied
subfolders_to_copy = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism', 'NormalVideos']

# Create 'train_sep' and 'test_sep' directories if they don't exist
os.makedirs(train_sep_dir, exist_ok=True)
os.makedirs(test_sep_dir, exist_ok=True)

# Function to copy first n files from source directory to destination directory
def copy_first_n_files(src_dir, dest_dir, n):
    os.makedirs(dest_dir, exist_ok=True)
    files = os.listdir(src_dir)[:n]
    for file in files:
        src_file = os.path.join(src_dir, file)
        dest_file = os.path.join(dest_dir, file)
        shutil.copy(src_file, dest_file)

# Copy first 10000 files from each subfolder in 'train' directory to 'train_sep' directory
for subfolder in subfolders_to_copy:
    src_folder = os.path.join(train_source_dir, subfolder)
    dest_folder = os.path.join(train_sep_dir, subfolder)
    copy_first_n_files(src_folder, dest_folder, 10000)

# Copy first 2500 files from each subfolder in 'test' directory to 'test_sep' directory
for subfolder in subfolders_to_copy:
    src_folder = os.path.join(test_source_dir, subfolder)
    dest_folder = os.path.join(test_sep_dir, subfolder)
    copy_first_n_files(src_folder, dest_folder, 2500)

# Copy first 10000 files from 'normal' subfolder in 'train' directory to 'train_sep' directory
src_normal_folder = os.path.join(train_source_dir, 'NormalVideos')
dest_normal_folder = os.path.join(train_sep_dir, 'NormalVideos')
os.makedirs(dest_normal_folder, exist_ok=True)
normal_files = os.listdir(src_normal_folder)[:10000]
for file in normal_files:
    src_file = os.path.join(src_normal_folder, file)
    dest_file = os.path.join(dest_normal_folder, file)
    shutil.copy(src_file, dest_file)

# Copy first 2500 files from 'normal' subfolder in 'test' directory to 'test_sep' directory
src_normal_folder = os.path.join(test_source_dir, 'NormalVideos')
dest_normal_folder = os.path.join(test_sep_dir, 'NormalVideos')
os.makedirs(dest_normal_folder, exist_ok=True)
normal_files = os.listdir(src_normal_folder)[:2500]
for file in normal_files:
    src_file = os.path.join(src_normal_folder, file)
    dest_file = os.path.join(dest_normal_folder, file)
    shutil.copy(src_file, dest_file)


train_generator = train_datagen.flow_from_directory(
        "./train_sep",
        target_size=(64, 64),
        batch_size=32,
        class_mode='sparse')

test_generator = test_datagen.flow_from_directory(
        "./test_sep",
        target_size=(64, 64),
        batch_size=32,
        class_mode='sparse')

# Model Architecture
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='gelu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='gelu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='gelu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='gelu'),
    layers.Dense(1, activation='sigmoid')
])


# input_shape = (64, 64, 3)
# inputs = Input(shape=input_shape)
# x = layers.Conv2D(64, (3, 3), activation='gelu')(inputs)
# x = layers.MaxPooling2D((2, 2))(x)
# x = layers.Conv2D(128, (3, 3), activation='gelu')(x)
# x = layers.MaxPooling2D((2, 2))(x)
# x = layers.Conv2D(256, (3, 3), activation='gelu')(x)
# x = layers.MaxPooling2D((2, 2))(x)
# x = layers.Flatten()(x)
# x = layers.Dense(256, activation='gelu')(x)
# outputs = layers.Dense(1, activation='sigmoid')(x)

# model = models.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Model Training
model.fit(train_generator, epochs=4, validation_data=test_generator)


# Save the trained model
model.save('model.h5')  # Lưu mô hình vào file 'model.h5'

# Model Evaluation
test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)

# Assuming y_true and y_pred are the true labels and predicted labels respectively
# Generate predictions using the trained model
y_true = []
y_pred = []
y_score = []

for i in range(len(test_generator)):
    #print('test generator is: ', test_generator)
    X_batch, y_batch = test_generator[i]
    #print('X_batch is: ',X_batch)
    #print('real y_batch is: ',y_batch)
    y_true.extend(y_batch)
    y_pred_batch = model.predict(X_batch)
    #print('predicted y_batch is: ',y_pred_batch)
    y_pred.extend((y_pred_batch > 0.5).astype(int).flatten())
    #print('y_pred is: ',y_pred)
    y_score.extend(y_pred_batch.flatten())
    #print('y_score is: ',y_score)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_score = np.array(y_score)

#print('y_true is: ',y_true)
#print('y_pred is: ',y_pred)
#print('y_score is: ',y_score)
# Log Loss
log_loss_value = log_loss(y_true, y_score)
print(f'Log Loss: {log_loss_value}')

# Sensitivity (Recall for the positive class)
sensitivity = recall_score(y_true, y_pred, pos_label=1)
print(f'Sensitivity: {sensitivity}')


# Mean Reciprocal Rank (MRR)
def mean_reciprocal_rank(y_true, y_score):
    # Convert inputs to numpy arrays for easier manipulation
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Initialize the list to store the reciprocal ranks
    reciprocal_ranks = []

    # Iterate over each true value
    for i in range(len(y_true)):
        if y_true[i] == 1:
            # Get the indices of the scores sorted in descending order
            sorted_indices = np.argsort(y_score)[::-1]

            # Find the rank of the current true positive in the sorted scores
            rank = np.where(sorted_indices == i)[0][0] + 1  # Adding 1 because rank starts from 1

            # Compute the reciprocal of the rank
            reciprocal_ranks.append(1 / rank)

    # Calculate the mean reciprocal rank
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
    return mrr


mrr = mean_reciprocal_rank(y_true, y_score)
print(f'Mean Reciprocal Rank (MRR): {mrr}')


# Discounted Cumulative Gain (DCG)
def dcg_at_k(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true_sorted = np.take(y_true, order[:k])
    gain = 2 ** y_true_sorted - 1
    discounts = np.log2(np.arange(len(y_true_sorted)) + 2)
    return np.sum(gain / discounts)


dcg = dcg_at_k(y_true, y_score)
print(f'Discounted Cumulative Gain (DCG): {dcg}')


# Normalized Discounted Cumulative Gain (NDCG)
def ndcg_at_k(y_true, y_score, k=10):
    dcg_max = dcg_at_k(y_true, y_true, k)
    if not dcg_max:
        return 0.
    return dcg_at_k(y_true, y_score, k) / dcg_max


ndcg = ndcg_at_k(y_true, y_score)
print(f'Normalized Discounted Cumulative Gain (NDCG): {ndcg}')


# Intersection over Union (IoU)
def intersection_over_union(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


iou = intersection_over_union(y_true, y_pred)
print(f'Intersection over Union (IoU): {iou}')


# Normalized Absolute Error (NAE)
def normalized_absolute_error(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


nae = normalized_absolute_error(y_true, y_pred)
print(f'Normalized Absolute Error (NAE): {nae}')