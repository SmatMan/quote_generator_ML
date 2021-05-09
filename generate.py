from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import os
import datetime
from tensorflow.python.client import device_lib
import random
import sys
import string

seedList = ["The ", "Life is like ", "I think that ", "If you think about it ", "Love is ", "I can't stress enough ", "I ", "Me ", "We ", "Did you know "]

text = open('datasets/quotes.txt', 'rb').read().decode(encoding='utf-8')

vocab = sorted(set(text))
print ('There are {} unique characters'.format(len(vocab)))
char2int = {c:i for i, c in enumerate(vocab)}
int2char = np.array(vocab)
print('Vector:\n')
for char,_ in zip(char2int, range(len(vocab))):
    print(' {:4s}: {:3d},'.format(repr(char), char2int[char]))

text_as_int = np.array([char2int[ch] for ch in text], dtype=np.int32)
print ('{}\n mapped to integers:\n {}'.format(repr(text[:100]), text_as_int[:100]))

tr_text = text_as_int[:704000] 
val_text = text_as_int[704000:] 
print(text_as_int.shape, tr_text.shape, val_text.shape)

batch_size = 64
buffer_size = 10000
embedding_dim = 256
epochs = 50
seq_length = 200
examples_per_epoch = len(text)//seq_length
rnn_units = 1024
vocab_size = len(vocab)

tr_char_dataset = tf.data.Dataset.from_tensor_slices(tr_text)
val_char_dataset = tf.data.Dataset.from_tensor_slices(val_text)
tr_sequences = tr_char_dataset.batch(seq_length+1, drop_remainder=True)
val_sequences = val_char_dataset.batch(seq_length+1, drop_remainder=True)
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text
tr_dataset = tr_sequences.map(split_input_target).shuffle(buffer_size).batch(batch_size, drop_remainder=True)
val_dataset = val_sequences.map(split_input_target).shuffle(buffer_size).batch(batch_size, drop_remainder=True)
print(tr_dataset, val_dataset)

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
     tf.keras.layers.Embedding(vocab_size, embedding_dim,
     batch_input_shape=[batch_size, None]),
     tf.keras.layers.Dropout(0.2),
     tf.keras.layers.LSTM(rnn_units,
     return_sequences=True,
     stateful=True,
     recurrent_initializer='glorot_uniform'),
     tf.keras.layers.Dropout(0.2), 
     tf.keras.layers.LSTM(rnn_units,
     return_sequences=True,
     stateful=True,
     recurrent_initializer='glorot_uniform'),
     tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(vocab_size)
 ])
    return model

checkpoint_dir = './checkpoints/'

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir)) 
model.build(tf.TensorShape([1, None]))
def generate_text(model, start_string):
    num_generate = 100
    input_eval = [char2int[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    temperature = 1.0
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions,      num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(int2char[predicted_id])
        if int2char[predicted_id] == ".":
            break 


    return (start_string + ''.join(text_generated))
    
print("Would you like to use the pre-generated seed list or specifiy your own seed?")
print("1) Use pre-generated seed list")
print("2) Use your own seed")
checkType = input(": ")

if checkType == "1":
    generatedSeed = random.choice(seedList)
    output = generate_text(model, generatedSeed)
elif checkType == "2":
    generatedSeed = input("Specify a seed: ")
    output = generate_text(model, generatedSeed)
else:
    print("Not an option.")
    sys.exit()

printable = set(string.printable)
print(''.join(filter(lambda x: x in printable, output)))