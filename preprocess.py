import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

# Load the CSV file containing metadata about the images and numerical values
data = pd.read_csv('data.csv')

# Preprocess images
image_size = (224, 224)  # Define the target size for images (224x224 for VGG16)

def preprocess_images(filenames, directory):
    images = []
    for file in filenames:
        img = load_img(f'{directory}/{file}', target_size=image_size)  # Load image and resize it
        img = img_to_array(img)  # Convert image to array
        images.append(img)
    return np.array(images)

# Group images by OD value, color, and consumable status
grouped_data = defaultdict(lambda: {'food_filenames': [], 'tube_filenames': []})

for index, row in data.iterrows():
    key = (row['OD_value'], row['color'], row['consumable'])
    grouped_data[key]['food_filenames'].append(row['food_filename'])
    grouped_data[key]['tube_filenames'].append(row['test_tube_filename'])

food_images = []
test_tube_images = []
numerical_data = []
labels = []

for (OD_value, color, consumable), filenames in grouped_data.items():
    food_imgs = preprocess_images(filenames['food_filenames'], 'food_images')
    tube_imgs = preprocess_images(filenames['tube_filenames'], 'test_tube_images')
    food_images.append(food_imgs)
    test_tube_images.append(tube_imgs)
    num_data = np.array([OD_value, color])
    numerical_data.append(np.tile(num_data, (food_imgs.shape[0], 1)))
    labels.append(np.array([1 if consumable == 'consumable' else 0] * food_imgs.shape[0]))

food_images = np.concatenate(food_images, axis=0)
test_tube_images = np.concatenate(test_tube_images, axis=0)
numerical_data = np.concatenate(numerical_data, axis=0)
labels = np.concatenate(labels, axis=0)

# Scale numerical data
scaler = StandardScaler()
numerical_data = scaler.fit_transform(numerical_data)

# Split the data into training and testing sets
X_train_food, X_test_food, X_train_tube, X_test_tube, X_train_num, X_test_num, y_train, y_test = train_test_split(
    food_images, test_tube_images, numerical_data, labels, test_size=0.2, random_state=42
)

# Save the preprocessed data to disk
np.savez('preprocessed_data.npz',
         X_train_food=X_train_food, X_test_food=X_test_food,
         X_train_tube=X_train_tube, X_test_tube=X_test_tube,
         X_train_num=X_train_num, X_test_num=X_test_num,
         y_train=y_train, y_test=y_test, scaler=scaler)
