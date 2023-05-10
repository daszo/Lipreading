import os
import cv2
import tensorflow as tf
from typing import List


vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

# Create lookup tables for char <-> num
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
    )


def load_video(path:str) -> List[float]:
    """ Loads video from path and returns normalized frames."""

    cap = cv2.VideoCapture(path)

    # Read video 
    # Isolate mouth region and add the frames to list
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)

        # Isolate mouth region. I opted to make it static, the LipNet paper used DLib.
        frames.append(frame[190:236,80:220,:])
    cap.release()
   
    # Normalize frames
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std

def load_alignments(path:str) -> List[str]:
    """ Loads alignments from path and returns a list of tokens."""

    with open(path, 'r') as f: 
        lines = f.readlines() 

    # Remove 'sil' (silence) tokens and position information. Add the rest to a list.
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil': 
            tokens = [*tokens,' ',line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

def load_data(path: str): 
    """ Loads video and alignments from path and returns a tuple of tensors."""

    # Convert path to video and alignment path
    path = bytes.decode(path.numpy())
    file_name = path.split('\\')[-1].split('.')[0]
    video_path = "D:\\python program\\lipreading\\" + os.path.join('data','s1',f'{file_name}.mpg')
    alignment_path = "D:\\python program\\lipreading\\" + os.path.join('data','alignments','s1',f'{file_name}.align')

    # Load video and alignments
    frames = load_video(video_path) 
    alignments = load_alignments(alignment_path)
    
    return frames, alignments

def mappable_function(path:str) ->List[str]:
    """ Wrap load_path in a tf.pyfuction. This also allows for GPU utilisation."""

    result = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    return result

def scheduler(epoch, lr):
    """ Learning rate scheduler. Decreases learning rate after 30 epochs."""
    
    if epoch < 30:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
    
def CTCLoss(y_true, y_pred):
    """ Custom CTC loss function which I copied form here: https://keras.io/examples/audio/ctc_asr/#model"""
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

class ProduceExample(tf.keras.callbacks.Callback): 
    """ Custom callback to print an example after each epoch."""
    def __init__(self, dataset) -> None: 
        self.dataset = dataset.as_numpy_iterator()
    
    def on_epoch_end(self, epoch, logs=None) -> None:
        data = self.dataset.next()
        yhat = self.model.predict(data[0])
        decoded = tf.keras.backend.ctc_decode(yhat, [75,75], greedy=False)[0][0].numpy()
        for x in range(len(yhat)):           
            print('Original:', tf.strings.reduce_join(num_to_char(data[1][x])).numpy().decode('utf-8'))
            print('Prediction:', tf.strings.reduce_join(num_to_char(decoded[x])).numpy().decode('utf-8'))
            print('~'*100)