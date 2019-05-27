import numpy as np
import os
from keras.models import Model
from keras.layers import Embedding, Dense, Input, Lambda, GRU, Bidirectional, Flatten
from keras.backend import softmax, concatenate, int_shape, reshape, batch_dot
from keras import regularizers
from keras.optimizers import RMSprop


def build_model(tokenizer):
    embedding_weights = load_pretrained_embedding(
        '/home/aftaab/Datasets/glove.6B', 50, tokenizer)

    # Model layers
    question = Input(shape=(22,))
    context = Input(shape=(325,))
    embedding = Embedding(len(tokenizer.word_index) + 2, 50, weights=[embedding_weights], trainable=False)
    question_encoder = Bidirectional(GRU(200, input_shape=(22, 50), return_sequences=True, recurrent_dropout=0.2))
    context_encoder = Bidirectional(GRU(200, input_shape=(325, 50), return_sequences=True, recurrent_dropout=0.4))
    attention = Lambda(lambda x: softmax(x))
    attention_in = Lambda(lambda x: batch_dot(x[0], x[1]))
    attention_matrix = Lambda(lambda x: batch_dot(x[0], x[1]))
    concat_layers = Lambda(lambda x: concatenate([x[0], x[1]], axis=-1))
    flatten = Flatten()
    output_start = Dense(325, activation='softmax', kernel_regularizer=regularizers.l2(0.01),
                         bias_regularizer=regularizers.l2(0.01), name='p_start')
    output_end = Dense(325, activation='softmax', kernel_regularizer=regularizers.l2(0.01),
                       bias_regularizer=regularizers.l2(0.01), name='p_end')

    # Since using sigmoid and binary_crossentropy along with categorical will require a custom loss function,
    # I've converted the output to a categorical variable([0, 1] for no ans, [1, 0] otherwise), and used one
    # loss function for the entire model
    output_no_answer = Dense(2, activation='softmax', name='no_answer')

    # Model architecture
    question_embedding = embedding(question)
    context_embedding = embedding(context)
    question_encoder_output = question_encoder(question_embedding)
    context_encoder_output = context_encoder(context_embedding)

    # To take a transpose of context_encoder_output, while keeping the batch dimension fixed
    # TODO: if you know of a cleaner way to transpose a (None, d1, d2) to (None, d2, d1), please send a PR
    batch_size, dim1, dim2 = int_shape(question_encoder_output)
    in_shape = (dim2, dim1)
    in_shape = (-1,) + in_shape

    question_encoder_out = Lambda(lambda x: reshape(x, in_shape))(question_encoder_output)
    attention_in = attention_in([context_encoder_output, question_encoder_out])
    attention_out = attention(attention_in)  # 1000 x 50
    attention_matrix = attention_matrix([attention_out, question_encoder_output])  # 1000 x 100
    concatenated = concat_layers([context_encoder_output, attention_matrix])
    concatenated = flatten(concatenated)

    out_start = output_start(concatenated)
    out_noans = output_no_answer(concatenated)
    out_end = output_end(concatenated)

    model = Model(inputs=[question, context], outputs=[out_start, out_end, out_noans])
    model.summary()
    model.compile(optimizer=RMSprop(0.00001), loss='categorical_crossentropy', metrics=['acc'])
    return model


def load_pretrained_embedding(glove_dir, embedding_dim, tokenizer):
    embeddings_index = {}
    with open(os.path.join(glove_dir, 'glove.6B.50d.txt'), 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print('Found %s word vectors.' % len(embeddings_index))
    embedding_matrix = np.zeros([len(tokenizer.word_index) + 2, embedding_dim], dtype="float32")
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
