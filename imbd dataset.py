# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 17:11:28 2020

@author: yash
"""

import tensorflow_datasets as tfds

imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
import numpy as np

train_data, test_data = imdb['train'], imdb['test']

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

# str(s.tonumpy()) is needed in Python3 instead of just s.numpy()
for s, l in train_data:
    training_sentences.append(s.numpy().decode('utf8'))  # copy imdb train_data to training_sentences
    training_labels.append(l.numpy())

for s, l in test_data:
    testing_sentences.append(s.numpy().decode('utf8'))
    testing_labels.append(l.numpy())

training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
oov_tok = "<OOV>"

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)  # tokenizer fit on training sentences
word_index = tokenizer.word_index  # gives word and their index
sequences = tokenizer.texts_to_sequences(
    training_sentences)  # gives dictionary with word index and the word in a sentence
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)  # padding is done

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length)

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])  #reverses the key value pair to word index:word

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(padded[1]))     #prints the decoded review by tokenizer
print(training_sentences[1])        #prints the actual review

import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    #tf.keras.layers.Conv1D(128, 5, activation='relu'),  #these could be used  and accuracy would have been 0.9933 and validation accuracy=0.8514
    #tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Flatten(),  #output of embedding will be flattened
    tf.keras.layers.Dense(6, activation='relu'),   #then it will be fed into a dense layer with 6 neuron
    tf.keras.layers.Dense(1, activation='sigmoid')  #then again to a dense layer with sigmoid activation function with only 1 neuron
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

num_epochs = 10
model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final)) #the neural network is trained with 25000 training examples and tested with 25000 examples for 10 times(epoch)

e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size=10000, embedding_dim=16)

import io

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')  #vecs.tsv and meta.tsv is loaded
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()

try:
  from google.colab import files
except ImportError:
  pass
else:
  files.download('vecs.tsv')
  files.download('meta.tsv')

sentence = "I really think this is amazing. honest."
sequence = tokenizer.texts_to_sequences([sentence])
print(sequence)

#http://projector.tensorflow.org/ go to this for embedding projection