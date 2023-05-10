# Authors:  Daniel van Oosteroom Date: 10/05/2023
import os
import tensorflow as tf

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, TimeDistributed, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

from utils import *

""" This model is based on the LipNet paper: https://github.com/rizkiarm/LipNet. ALso see: https://arxiv.org/pdf/1611.01599.pdf
LipNet is a model for lipreading. It takes a video of a person speaking as input and outputs a sequence of characters.
I use a small part of the LipNet dataset. The model is a 3D CNN followed by 2 layers of Bi LSTM. The model is trained using CTC loss.
In the future I would like to make predicitons with this model. I was not able to get to that becasue I got stuck on the GPU part."""

# Set up GPU  and remove any ussage of GPU memory.
# This was the part that I got stuck on. Tensorflow did not recognize my GPU. 
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

def main():
    
    # Create Data Pipeline
    # I would have liked to use relative paths, but I could not get it to work.
    data = tf.data.Dataset.list_files('D:\python program\lipreading\data\s1\*.mpg')  
    data = data.shuffle(500, reshuffle_each_iteration=False)
    data = data.map(mappable_function)
    data = data.padded_batch(2, padded_shapes=([75,None,None,None],[40]))

    # Overlap preprocessing and training
    data = data.prefetch(tf.data.AUTOTUNE)

    # Make split, 
    train = data.take(450)
    test = data.skip(450)

    # Create Model
    model = Sequential()
    model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same', activation='relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(256, 3, padding='same', activation='relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(75, 3, padding='same', activation='relu'))

    model.add(TimeDistributed(Flatten()))

    # Add LSTM layers, in the future this can be replaced with a transformer
    # The LipNet paper used 2 layers of BiGRU 
    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Dense(char_to_num.vocabulary_size()+1, kernel_initializer='he_normal', activation='softmax'))

    # Set up training
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=CTCLoss)
    checkpoint_callback = ModelCheckpoint(os.path.join('models','checkpoint'), monitor='loss', save_weights_only=True) 
    schedule_callback = LearningRateScheduler(scheduler)
    example_callback = ProduceExample(test)

    # Train
    model.fit(train, validation_data=test, epochs=100, callbacks=[checkpoint_callback, schedule_callback, example_callback])


if __name__ == '__main__':
    main()