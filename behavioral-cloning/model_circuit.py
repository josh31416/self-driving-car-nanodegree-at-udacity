import pandas as pd
import numpy as np

driving_log = pd.read_csv('circuit_data/driving_log.csv')

correction = 0.15
driving_log['steering_left'] = driving_log['steering'] + correction
driving_log['steering_right'] = driving_log['steering'] - correction

df = pd.DataFrame(pd.concat([pd.concat([driving_log['center'].str.strip(),
                                        driving_log['left'].str.strip(),
                                        driving_log['right'].str.strip()], ignore_index=True),
                             pd.concat([driving_log['steering'],
                                        driving_log['steering_left'],
                                        driving_log['steering_right']], ignore_index=True)
                            ], axis=1, keys=['img_path', 'steering']))


from keras.preprocessing.image import ImageDataGenerator

data_generator = ImageDataGenerator(brightness_range=(0.2, 1),
                                    height_shift_range=0.1,
                                    rotation_range=5,
                                    zoom_range=0.2,
                                    channel_shift_range=150.0)\
    .flow_from_dataframe(df,
                         directory='circuit_data',
                         x_col='img_path', y_col='steering',
                         target_size=(160, 320),
                         color_mode='rgb',
                         class_mode='raw',
                         batch_size=128,
                         shuffle=True)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout
from keras.optimizers import Adam

model = Sequential()

# Crop undesired parts of the image
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
# Normalize the images
model.add(Lambda(lambda x: x / 255.0 - 0.5))

model.add(Conv2D(24, 5, strides=(2, 2), activation='relu'))
model.add(Conv2D(36, 5, strides=(2, 2), activation='relu'))
model.add(Conv2D(48, 5, strides=(2, 2), activation='relu'))
model.add(Conv2D(64, 1, activation='relu'))
model.add(Conv2D(64, 3, activation='relu'))

model.add(Flatten())
model.add(Dropout(.5))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

adam = Adam(lr=0.0001)
model.compile(loss='mse', optimizer=adam)

model.fit_generator(generator=data_generator, epochs=5, verbose = 1, workers=8)
model.save('model_circuit.h5')