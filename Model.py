import numpy as np
import os
from keras.models import Model
from keras.layers import Embedding, Dense, Input, LSTM, Activation, Lambda
import keras.backend as K

def keras_focal_loss(target, pred):
	gamma = 2.
	pred = K.cast(pred, np.float32)
	max_val = K.clip(-pred, 0, 1)
	loss = pred - pred * target + max_val + K.log(K.exp(-max_val) + K.exp(-pred - max_val))
	invprobs = K.log(K.sigmoid(-pred * (target * 2.0 - 1.0)))
	loss = K.exp(invprobs * gamma) * loss
	return K.mean(K.sum(loss, axis=1))


def build_model(tokenizer):
	embedding_weights = load_pretrained_embedding(
		'/home/aftaab/Datasets/GloveEmbeddings/glove.6D', 300, tokenizer)

	# Model layers
	question = Input(shape=(50,))
	context = Input(shape=(1000,))
	embedding = Embedding(len(tokenizer.word_index) + 2, 300, weights=[embedding_weights], trainable=False)
	question_encoder = LSTM(64, input_shape=(None, 50, 300))
	context_encoder = LSTM(64, input_shape=(None, 1000, 300))
	attention = Lambda(lambda x: x[0] * x[1])
	attention_scores = Activation('softmax')
	output = Dense(1000, activation='sigmoid')

	# Model architecture
	emb_q = embedding(question)
	emb_c = embedding(context)
	q_enc_out = question_encoder(emb_q)
	c_enc_out = context_encoder(emb_c)
	attention = attention([q_enc_out, c_enc_out])
	attention_scores = attention_scores(attention)
	out = output(attention_scores)
	model = Model(inputs=[question, context], outputs=[out])
	model.summary()
	model.compile(optimizer='adadelta', loss=keras_focal_loss, metrics=[keras_focal_loss])
	return model


def load_pretrained_embedding(glove_dir, embedding_dim, tokenizer):
	embeddings_index = {}
	with open(os.path.join(glove_dir, 'glove.6B.300d.txt'), 'r') as f:
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
