from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences


class Preprocessing:

	def __init__(self, size):
		self.tokenizer = Tokenizer(oov_token='<UNK>', num_words=size + 1, char_level=False)
		self.vocab_size = 0
		self.num_words = size

	def create_vocabulary(self, texts):
		self.tokenizer.fit_on_texts(texts)
		self.tokenizer.word_index = {e: i for e, i in self.tokenizer.word_index.items() if i <= self.num_words}
		self.tokenizer.index_word = {i: e for e, i in self.tokenizer.word_index.items() if i <= self.num_words}
		self.tokenizer.word_index[self.tokenizer.oov_token] = self.num_words + 1
		self.vocab_size = self.num_words + 1

	def text_to_seq(self, text, maxlen):
		seq = []
		words = text_to_word_sequence(text)
		for word in words:
			if self.tokenizer.word_index.get(word):
				seq.append(self.tokenizer.word_index[word])
			else:
				seq.append(self.tokenizer.word_index['<UNK>'])
		seq = pad_sequences([seq], maxlen=maxlen, truncating='post', padding='post')
		return seq

	def text_to_seq_sample(self, sample, seq_len=(450, 15, 15)):
		sequence = list()
		sequence.append(self.text_to_seq(sample[0], seq_len[0]))
		sequence.append(self.text_to_seq(sample[1], seq_len[1]))
		sequence.append(self.text_to_seq(sample[2], seq_len[2]))
		return sequence

	def sequence_to_text(self, sequence):
		text = ''
		for seq in sequence:
			for s in seq:
				if self.tokenizer.index_word.get(s):
					text = text + ' ' + self.tokenizer.index_word[s]
		return text.strip()
