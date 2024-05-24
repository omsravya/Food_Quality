from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load preprocessed data
data = np.load('preprocessed_data.npz')
X_test_food = data['X_test_food']
X_test_tube = data['X_test_tube']
X_test_num = data['X_test_num']
y_test = data['y_test']

# Normalize image data
X_test_food = preprocess_input(X_test_food)
X_test_tube = preprocess_input(X_test_tube)

# Load the trained model
model = load_model('food_model.h5')

# Evaluate the model on the test data
loss, accuracy = model.evaluate([X_test_food, X_test_tube, X_test_num], y_test)
print(f'Test Accuracy: {accuracy:.2f}')
