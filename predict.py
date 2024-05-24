from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load the trained model
model = load_model('food_model.h5')

# Load the scaler
data = np.load('preprocessed_data.npz', allow_pickle=True)
scaler = data['scaler'].item()

# Preprocess new food image
image_size = (224, 224)
new_food_image = load_img('new_food_image.jpg', target_size=image_size)
new_food_image = img_to_array(new_food_image)
new_food_image = preprocess_input(new_food_image)
new_food_image = np.expand_dims(new_food_image, axis=0)

# Preprocess new test tube image
new_tube_image = load_img('new_tube_image.jpg', target_size=image_size)
new_tube_image = img_to_array(new_tube_image)
new_tube_image = preprocess_input(new_tube_image)
new_tube_image = np.expand_dims(new_tube_image, axis=0)

# Preprocess new numerical data
new_OD_value = 9  # Example value
new_color = 6  # Example value
new_numerical_data = scaler.transform([[new_OD_value, new_color]])

# Make a prediction
prediction = model.predict([new_food_image, new_tube_image, new_numerical_data])
is_consumable = prediction[0][0] > 0.2
print('Consumable' if is_consumable else 'Not Consumable')
print(prediction[0][0])
