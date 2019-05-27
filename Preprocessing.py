from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import params

class Preprocessing:

    def __init__(self, size):
        self.tokenizer = Tokenizer(oov_token='<UNK>', char_level=False)
        self.vocab_size = 0
        self.num_words = size

    def create_vocabulary(self, texts):
        self.tokenizer.fit_on_texts(texts)
        self.tokenizer.word_index = {e: i for e, i in self.tokenizer.word_index.items() if i <= self.num_words}
        self.tokenizer.index_word = {i: e for e, i in self.tokenizer.word_index.items() if i <= self.num_words}
        self.tokenizer.word_index[self.tokenizer.oov_token] = len(self.tokenizer.word_index) + 1
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

    def text_to_seq_sample(self, sample, seq_len=(params.CONTEXT_LEN, params.QUESTION_LEN)):
        sequence = list()
        sequence.append(self.text_to_seq(sample[0], seq_len[0]))
        sequence.append(self.text_to_seq(sample[1], seq_len[1]))
        target_start = np.zeros(shape=(params.CONTEXT_LEN,))
        target_end = np.zeros(shape=(params.CONTEXT_LEN,))
        sequence.append(sample[2])
        if not sample[3] == -1:
            target_start[sample[3]] = 1
            target_end[sample[4]] = 1
        sequence.append(target_start)
        sequence.append(target_end)
        sequence.append(np.array(sample[5]))
        if len(sample) == 7:
            sequence.append(sample[-1])
        return sequence

    def sequence_to_text(self, sequence):
        text = ''
        for s in sequence:
            if self.tokenizer.index_word.get(s):
                text = text + ' ' + self.tokenizer.index_word[s]
        return text.strip()

    def target_to_words(self, context, start, end, is_no_ans):
        if is_no_ans.argmax() == 1:
            return ''
        if start < end + 1:
            return self.sequence_to_text(context[start:end + 1])
        else:
            return ''
