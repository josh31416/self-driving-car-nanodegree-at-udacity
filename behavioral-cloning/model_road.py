import pandas as pd
import numpy as np

# Load CSV
driving_log = pd.read_csv('road_data/driving_log.csv')

# Use left and right images for recovering behavior
correction = 0.15
driving_log['steering_left'] = driving_log['steering'] + correction
driving_log['steering_right'] = driving_log['steering'] - correction

# Build pandas dataframe to feed to `flow_from_dataframe` function
df = pd.DataFrame(pd.concat([pd.concat([driving_log['center'].str.strip(),
                                        driving_log['left'].str.strip(),
                                        driving_log['right'].str.strip()], ignore_index=True),
                             pd.concat([driving_log['steering'],
                                        driving_log['steering_left'],
                                        driving_log['steering_right']], ignore_index=True)
                            ], axis=1, keys=['img_path', 'steering']))

# Import keras ImageDataGenerator to augment the data
from keras.preprocessing.image import ImageDataGenerator

data_generator = ImageDataGenerator(brightness_range=(0.1, 0.8),
                                    height_shift_range=0.1,
                                    rotation_range=5,
                                    zoom_range=0.2,
                                    channel_shift_range=150.0)\
    .flow_from_dataframe(df,
                         directory='road_data',
                         x_col='img_path', y_col='steering',
                         target_size=(160, 320),
                         color_mode='rgb',
                         class_mode='raw',
                         batch_size=128,
                         shuffle=True)

# Import keras dependencies
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout
from keras.optimizers import Adam

# Build the model
model = Sequential()
# Crop undesired parts of the image
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
# Normalize the images
model.add(Lambda(lambda x: x / 255.0 - 0.5))
# 5x5 conv layers
model.add(Conv2D(24, 5, strides=(2, 2), activation='relu'))
model.add(Conv2D(36, 5, strides=(2, 2), activation='relu'))
model.add(Conv2D(48, 5, strides=(2, 2), activation='relu'))
# 3x3 conv layers
model.add(Conv2D(64, 3, activation='relu'))
model.add(Conv2D(64, 3, activation='relu'))
# Fully connected
model.add(Flatten())
model.add(Dropout(.5)) # Regularization
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1)) # Steering output

# Adam optimizer
adam = Adam(lr=0.0001)
# Train with mean squared error loss and adam optimizer
model.compile(loss='mse', optimizer=adam)

# Train the model
history = model.fit_generator(generator=data_generator, epochs=10, verbose = 1, workers=8)
# Save the model
model.save('model_road.h5')

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()