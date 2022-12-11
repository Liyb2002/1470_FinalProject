import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Dense, Input, Embedding, Concatenate, TimeDistributed, Bidirectional, GRU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model
from rouge import Rouge
from Summarization.attention import AttentionLayer

class Seq2Seq(tf.keras.Model):
    def __init__(self, latent_dim, embedding_dim, maxlen_text, text_vocab_length, headline_vocab_length, ** kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.encoder = tf.keras.Sequential([
            Input(shape=(maxlen_text,)),
            Embedding(text_vocab_length, embedding_dim, trainable=True),
            LSTM(latent_dim, return_sequences=True, return_state=True,
                 dropout=0.3, recurrent_dropout=0.2),
            LSTM(latent_dim, return_sequences=True, return_state=True,
                 dropout=0.3, recurrent_dropout=0.2),
            LSTM(latent_dim, return_sequences=True, return_state=True,
                 dropout=0.3, recurrent_dropout=0.2)
        ])
        self.decoder = tf.keras.Sequential([
            Input(shape=(None,)),
            Embedding(headline_vocab_length, embedding_dim, trainable=True),
            LSTM(latent_dim, return_sequences=True, return_state=True,
                 dropout=0.3, recurrent_dropout=0.2)
        ])
        self.attention = AttentionLayer(name='attention_layer')
        self.concat = Concatenate(axis=-1, name='concat_layer')
        self.dense = TimeDistributed(
            Dense(headline_vocab_length, activation="softmax"))

    def call(self, encoder_inputs, decoder_inputs):
        encoder_ouputs, state_h, state_c = self.encoder(encoder_inputs)
        states = [state_h, state_c]
        decoder_outputs, decoder_fwd_state, decoder_bwd_state = self.decoder(
            decoder_inputs, initial_state=states)
        context_vector, attention_states = self.attention(
            [encoder_ouputs, decoder_outputs])
        decoder_concat_output = self.concat([context_vector, decoder_outputs])
        decoder_outputs = self.dense(decoder_concat_output)
        return decoder_outputs