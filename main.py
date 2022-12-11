import numpy as np
import pandas as pd

import re
import string
import csv
from sklearn.model_selection import train_test_split
from Seq2Seq.model import Seq2Seq
from attention import AttentionLayer
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Dense, Input, Embedding, Concatenate, TimeDistributed, Bidirectional, GRU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model
from rouge import Rouge

clean_df = pd.read_csv(
    './cleaned_data.csv')
#Tokenizing training set and test set
train_x, test_x, train_y, test_y = train_test_split(clean_df['text'], clean_df['summary'], test_size=0.1, random_state=0)
del clean_df
t_tokenizer = Tokenizer()
t_tokenizer.fit_on_texts(list(train_x))

thresh = 4
count = 0
total_count = 0
frequency = 0
total_frequency = 0

for key, value in t_tokenizer.word_counts.items():
    total_count += 1
    total_frequency += value
    if value < thresh:
        count += 1
        frequency += value
# print('% of rare words in vocabulary: ', (count/total_count)*100.0)
# print('Total Coverage of rare words: ', (frequency/total_frequency)*100.0)
# t_max_features = total_count - count
# print('Text Vocab: ', t_max_features)
s_tokenizer = Tokenizer()
s_tokenizer.fit_on_texts(list(train_y))

thresh = 6
count = 0
total_count = 0
frequency = 0
total_frequency = 0

for key, value in s_tokenizer.word_counts.items():
    total_count += 1
    total_frequency += value
    if value < thresh:
        count += 1
        frequency += value


maxlen_text = 800
maxlen_summ = 150

val_x = test_x
t_tokenizer = Tokenizer(num_words=maxlen_text)
t_tokenizer.fit_on_texts(list(train_x))
text_vocab_length = len(t_tokenizer.index_word) + 1

train_x = t_tokenizer.texts_to_sequences(train_x)
val_x = t_tokenizer.texts_to_sequences(val_x)
train_x = pad_sequences(train_x, maxlen=maxlen_text, padding='post')
val_x = pad_sequences(val_x, maxlen=maxlen_text, padding='post')
val_y = test_y

s_tokenizer = Tokenizer(num_words=maxlen_summ)
s_tokenizer.fit_on_texts(list(train_y))
headline_vocab_length = len(s_tokenizer.index_word) + 1

train_y = s_tokenizer.texts_to_sequences(train_y)
val_y = s_tokenizer.texts_to_sequences(val_y)
train_y = pad_sequences(train_y, maxlen=maxlen_summ, padding='post')
val_y = pad_sequences(val_y, maxlen=maxlen_summ, padding='post')

#get pretrained embedding
embeding_index = {}
embed_dim = 100
with open('./glove6b/glove.6B.100d.txt') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeding_index[word] = coefs

t_embed = np.zeros((maxlen_text, embed_dim))
for word, i in t_tokenizer.word_index.items():
    vec = embeding_index.get(word)
    if i < maxlen_text and vec is not None:
        t_embed[i] = vec

s_embed = np.zeros((maxlen_summ, embed_dim))
for word, i in s_tokenizer.word_index.items():
    vec = embeding_index.get(word)
    if i < maxlen_summ and vec is not None:
        s_embed[i] = vec
del embeding_index

latent_dim = 300
embedding_dim = 100

encoder_inputs = Input(shape=(maxlen_text,))
encoder_emb = Embedding(t_embed, embedding_dim, input_length=maxlen_text, weights = [t_embed],
                        trainable=False)(encoder_inputs)

encoder_lstm1 = LSTM(latent_dim, return_sequences=True,
                     return_state=True, dropout=0.3, recurrent_dropout=0.2)
encoder_output1, state_h1, state_c1 = encoder_lstm1(encoder_emb)

encoder_lstm2 = LSTM(latent_dim, return_sequences=True,
                     return_state=True, dropout=0.3, recurrent_dropout=0.2)
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

encoder_lstm = LSTM(latent_dim, return_sequences=True,
                    return_state=True, dropout=0.3, recurrent_dropout=0.2)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_output2)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None,))
decoder_emb = Embedding(s_embed, embedding_dim, input_length=maxlen_summ, weights = [s_embed],
                        trainable=False)(decoder_inputs)

decoder_lstm = LSTM(latent_dim, return_sequences=True,
                    return_state=True, dropout=0.3, recurrent_dropout=0.2)
decoder_outputs, decoder_fwd_state, decoder_bwd_state = decoder_lstm(
    decoder_emb, initial_state=encoder_states)

attn_layer = AttentionLayer(name='attention_layer')
attn_outputs, attn_states = attn_layer([encoder_outputs, decoder_outputs])

decoder_concat_outputs = Concatenate(
    axis=-1, name='concat_layer')([decoder_outputs, attn_outputs])

decoder_dense = TimeDistributed(
    Dense(headline_vocab_length, activation='softmax'))
decoder_outputs = decoder_dense(decoder_concat_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', mode='min', verbose=1, patience=2)
model.fit(train_x,
              train_y,
              batch_size=128,
              epochs=10,
              validation_data=(test_x, test_y)
              )
model.save('seq2seq.h5')


encoder_model = Model(inputs=encoder_inputs, outputs=[
                      encoder_outputs, state_h, state_c])

decoder_initial_state_a = Input(shape=(latent_dim,))
decoder_initial_state_c = Input(shape=(latent_dim,))
decoder_hidden_state = Input(shape=(maxlen_text, latent_dim))

decoder_out, decoder_state_a, decoder_state_c = decoder_lstm(
    decoder_emb, initial_state=[decoder_initial_state_a, decoder_initial_state_c])
attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state, decoder_out])
decoder_inf_concat_outputs = Concatenate(
    axis=-1, name='concat')([decoder_out, attn_out_inf])

decoder_final = decoder_dense(decoder_inf_concat_outputs)
decoder_model = Model([decoder_inputs]+[decoder_hidden_state, decoder_initial_state_a,
                                        decoder_initial_state_c], [decoder_final]+[decoder_state_a, decoder_state_c])


def decode_sequences(input_sequence):
    encoder_out, encoder_a, encoder_c = encoder_model.predict(input_sequence)
    next_input = np.zeros((1, 1))
    next_input[0, 0] = t_tokenizer.word_index['start']
    output_sequence = ''
    stop = False
    while not stop:
        decoded_out, trans_state_a, trans_state_c = decoder_model.predict(
            [next_input] + [encoder_out, encoder_a, encoder_c])
        output_idx = np.argmax(decoded_out[0, -1, :])
        if output_idx == t_tokenizer.word_index['end']:
            stop = True
        elif output_idx > 0 and output_idx != t_tokenizer.word_index['start']:
            output_token = t_tokenizer.index_word[output_idx]
            output_sequence = output_sequence + ' ' + output_token
        next_input[0, 0] = output_idx
        # Continously update the transient state vectors in decoder.
        encoder_a, encoder_c = trans_state_a, trans_state_c

    return output_sequence


def sequence_to_text(input_sequence, mode):
    res = ''

    if mode == 'input':
        for idx in input_sequence:
            if idx:
                res = res + s_tokenizer.index_word[idx] + ' '

    elif mode == 'output':
        for idx in input_sequence:
            if idx:
                if s_tokenizer.index_word[idx] != 'start' and s_tokenizer.index_word[idx] != 'end':
                    res = res + s_tokenizer.index_word[idx] + ' '

    return res


predicted_summaries = []

for i in range(20):
    print("News Article:", sequence_to_text(test_x[i], 'input'))
    print("Original Article Summary:", sequence_to_text(test_y[i], 'output'))
    pred_summary = decode_sequences(test_x[i].reshape(1, maxlen_text))
    print("Predicted Article Summary:", pred_summary)
    predicted_summaries.append(pred_summary)
    print()
    print('---------------------------')
    print()
