from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.optimizers import Adam
from multi_input_data_generator import MultiInputDataGenerator
import numpy as np

# Load preprocessed data
data = np.load('preprocessed_data.npz')
X_train_food = data['X_train_food']
X_test_food = data['X_test_food']
X_train_tube = data['X_train_tube']
X_test_tube = data['X_test_tube']
X_train_num = data['X_train_num']
X_test_num = data['X_test_num']
y_train = data['y_train']
y_test = data['y_test']

# Normalize image data
X_train_food = preprocess_input(X_train_food)
X_test_food = preprocess_input(X_test_food)
X_train_tube = preprocess_input(X_train_tube)
X_test_tube = preprocess_input(X_test_tube)

# Define the model using VGG16 for food and test tube images
def create_vgg16_model(input_shape):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model = Model(inputs=base_model.input, outputs=x)
    for layer in base_model.layers:
        layer.trainable = False
    return model

# Food image input
food_input = Input(shape=(224, 224, 3))
food_model = create_vgg16_model((224, 224, 3))
food_features = food_model(food_input)

# Test tube image input
tube_input = Input(shape=(224, 224, 3))
tube_model = create_vgg16_model((224, 224, 3))
tube_features = tube_model(tube_input)

# Numerical input
num_input = Input(shape=(2,))
num_features = Dense(64, activation='relu')(num_input)

# Concatenate all inputs
combined = Concatenate()([food_features, tube_features, num_features])
z = Dense(64, activation='relu')(combined)
z = Dropout(0.5)(z)
z = Dense(1, activation='sigmoid')(z)

# Create the final model
model = Model(inputs=[food_input, tube_input, num_input], outputs=z)

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Create data generators
train_generator = MultiInputDataGenerator(X_train_food, X_train_tube, X_train_num, y_train, batch_size=32, shuffle=True)
validation_generator = MultiInputDataGenerator(X_test_food, X_test_tube, X_test_num, y_test, batch_size=32, shuffle=False)

# Train the model
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=50,
)

# Save the trained model
model.save('food_model.h5')
