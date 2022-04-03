import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Concatenate, TimeDistributed
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras.layers import Attention, AdditiveAttention

class EncoderDecoderAttention(object):
    '''
    Text summerization using the encoder-decoder model with an attention layer.

    '''

    def __init__(self, max_features, maxlength_txt, maxlength_sum, embedding_dim, num_class, layer_dim):
        self.max_features = max_features          # input dim for embedding
        self.maxlength_txt = maxlength_txt        # length of input sequence
        self.maxlength_sum = maxlength_sum        # length of summary
        self.embedding_dim = embedding_dim        # output dim for embedding
        self.num_class = num_class                # output dim of classes
        self.layer_dim = layer_dim
        self.model, self.encoder, self.decoder = self.model()
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
  

    def model(self):
        encoder_inputs = Input(shape=(self.maxlength_txt,))

        encoder_embedd_layer = Embedding(self.max_features, self.embedding_dim, input_length=self.maxlength_txt, trainable=True)
        encoder_embedd = encoder_embedd_layer(encoder_inputs)
        encoder = LSTM(self.layer_dim, return_sequences=True, return_state=True)
        encoder_output, state_h, state_c = encoder(encoder_embedd)
        encoder_state = [state_h, state_c]

        decoder_inputs = Input(shape=(None,))
        decoder_embedd_layer = Embedding(self.max_features, self.embedding_dim, input_length=self.maxlength_sum, trainable=True)
        decoder_embedd = decoder_embedd_layer(decoder_inputs)
        decoder = LSTM(self.layer_dim, return_sequences=True, return_state=True)
        decoder_output, _ , _ = decoder(decoder_embedd, initial_state=encoder_state)

        attn_output = Attention()([decoder_output, encoder_output]) 
        decoder_output = Concatenate(axis=-1)([decoder_output, attn_output])

        decoder_dense = TimeDistributed(Dense(self.num_class, activation='softmax'))
        decoder_output = decoder_dense(decoder_output)
        input_state = [encoder_inputs, decoder_inputs]
        model = Model(inputs = input_state, outputs = decoder_output)

        encoder_model = Model(encoder_inputs, outputs=[encoder_output, state_h, state_c])
        decoder_input_state = [Input(shape=(self.layer_dim,)), Input(shape=(self.layer_dim,))]
        decoder_hidden_state_input = Input(shape=(self.maxlength_txt, self.layer_dim))

        decoder_embedd2 = decoder_embedd_layer(decoder_inputs)
        decoder_output2, decostate_h2, decostate_c2 = decoder(decoder_embedd2, initial_state=decoder_input_state)

        attn_output_sum = Attention()([decoder_output2, decoder_hidden_state_input])
        decoder_output2 = Concatenate(axis=-1)([decoder_output2, attn_output_sum])

        decoder_output2 = decoder_dense(decoder_output2)
        decoder_model = Model([decoder_inputs]+[decoder_hidden_state_input]+decoder_input_state, 
                            [decoder_output2]+[decostate_h2, decostate_c2])

        return model, encoder_model, decoder_model


    def summarization(self, input_seq, word_index, index_word):
    
        encoder_output, encoder_h, enecoder_c = self.encoder.predict(input_seq)
        
        output_seq = np.zeros((1,1))
        output_seq[0,0] = word_index['ssstarttt']
        summary = ''
        for i in range(self.maxlength_sum):
            output_words, deco_h, deco_c = self.decoder.predict([output_seq] + [encoder_output, encoder_h, enecoder_c])
            sampled_word_idx = np.argmax(output_words[0, -1, :])

            sampled_word = index_word[sampled_word_idx]
            if sampled_word!='eeenddd':
                summary += (' ' + sampled_word)
            else:
                break
            output_seq = np.zeros((1,1))
            output_seq[0,0] = sampled_word_idx
            encoder_h, enecoder_c = deco_h, deco_c
        
        return summary


class EncoderDecoderAdditiveAttention(EncoderDecoderAttention):
    '''
      Text summarization using the encoder-decoder model with an additive attention layer.

    '''

    def __init__(self, *args):
        super().__init__(*args)

    def model(self):
        encoder_inputs = Input(shape=(self.maxlength_txt,))

        encoder_embedd_layer = Embedding(self.max_features, self.embedding_dim, input_length=self.maxlength_txt, trainable=True)
        encoder_embedd = encoder_embedd_layer(encoder_inputs)
        encoder = LSTM(self.layer_dim, return_sequences=True, return_state=True)
        encoder_output, state_h, state_c = encoder(encoder_embedd)
        encoder_state = [state_h, state_c]

        decoder_inputs = Input(shape=(None,))
        decoder_embedd_layer = Embedding(self.max_features, self.embedding_dim, input_length=self.maxlength_sum, trainable=True)
        decoder_embedd = decoder_embedd_layer(decoder_inputs)
        decoder = LSTM(self.layer_dim, return_sequences=True, return_state=True)
        decoder_output, _ , _ = decoder(decoder_embedd, initial_state=encoder_state)

        attn_output = AdditiveAttention()([decoder_output, encoder_output]) 
        decoder_output = Concatenate(axis=-1)([decoder_output, attn_output])

        decoder_dense = TimeDistributed(Dense(self.num_class, activation='softmax'))
        decoder_output = decoder_dense(decoder_output)
        input_state = [encoder_inputs, decoder_inputs]
        model = Model(inputs = input_state, outputs = decoder_output)

        encoder_model = Model(encoder_inputs, outputs=[encoder_output, state_h, state_c])
        decoder_input_state = [Input(shape=(self.layer_dim,)), Input(shape=(self.layer_dim,))]
        decoder_hidden_state_input = Input(shape=(self.maxlength_txt, self.layer_dim))

        decoder_embedd2 = decoder_embedd_layer(decoder_inputs)
        decoder_output2, decostate_h2, decostate_c2 = decoder(decoder_embedd2, initial_state=decoder_input_state)

        attn_output_sum = AdditiveAttention()([decoder_output2, decoder_hidden_state_input])
        decoder_output2 = Concatenate(axis=-1)([decoder_output2, attn_output_sum])

        decoder_output2 = decoder_dense(decoder_output2)
        decoder_model = Model([decoder_inputs]+[decoder_hidden_state_input]+decoder_input_state, 
                            [decoder_output2]+[decostate_h2, decostate_c2])

        return model, encoder_model, decoder_model

class BidirectEncoderDecoderAttention(EncoderDecoderAttention):
    '''
      Text summarization using the bidirectional encoder-decoder model with an attention layer.
    
    '''

    def __init__(self, *args):
        super().__init__(*args)

    def model(self):
        encoder_inputs = Input(shape=(self.maxlength_txt,))

        encoder_embedd_layer = Embedding(self.max_features, self.embedding_dim, input_length=self.maxlength_txt, trainable=True)
        encoder_embedd = encoder_embedd_layer(encoder_inputs)
        
        encoder = Bidirectional(LSTM(self.layer_dim, return_sequences=True, return_state=True), merge_mode="concat")
        encoder_output, state_h1, state_c1, state_h2, state_c2 = encoder(encoder_embedd)
        state_h = Concatenate()([state_h1, state_h2])
        state_c = Concatenate()([state_c1, state_c2])
        encoder_state = [state_h, state_c]

        decoder_inputs = Input(shape=(None,))
        decoder_embedd_layer = Embedding(self.max_features, self.embedding_dim, input_length=self.maxlength_sum, trainable=True)
        decoder_embedd = decoder_embedd_layer(decoder_inputs)
        decoder = LSTM(self.layer_dim*2, return_sequences=True, return_state=True)
        decoder_output, _ , _ = decoder(decoder_embedd, initial_state=encoder_state)

        attn_output = Attention()([decoder_output, encoder_output]) 
        decoder_output = Concatenate(axis=-1)([decoder_output, attn_output])

        decoder_dense = TimeDistributed(Dense(self.num_class, activation='softmax'))
        decoder_output = decoder_dense(decoder_output)
        input_state = [encoder_inputs, decoder_inputs]
        model = Model(inputs = input_state, outputs = decoder_output)

        encoder_model = Model(encoder_inputs, outputs=[encoder_output, state_h, state_c])
        decoder_input_state = [Input(shape=(self.layer_dim*2,)), Input(shape=(self.layer_dim*2,))]
        decoder_hidden_state_input = Input(shape=(self.maxlength_txt, self.layer_dim*2))

        decoder_embedd2 = decoder_embedd_layer(decoder_inputs)
        decoder_output2, decostate_h2, decostate_c2 = decoder(decoder_embedd2, initial_state=decoder_input_state)

        attn_output_sum = Attention()([decoder_output2, decoder_hidden_state_input])
        decoder_output2 = Concatenate(axis=-1)([decoder_output2, attn_output_sum])

        decoder_output2 = decoder_dense(decoder_output2)
        decoder_model = Model([decoder_inputs]+[decoder_hidden_state_input]+decoder_input_state, 
                            [decoder_output2]+[decostate_h2, decostate_c2])

        return model, encoder_model, decoder_model


