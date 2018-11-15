import json
import pickle
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer


def map_word_to_context(context_words, start_char_index):
	char_count = 0
	word_count = 0
	for word in context_words:
		if char_count == start_char_index:
			return word_count
		char_count += len(word)
		word_count += 1
	return None


def generate_dataset(type):
	if type == 'train':
		path = 'Dataset/train-v2.0.json'
	else:
		path = 'Dataset/dev-v2.0.json'

	result = []

	with open(path, 'r') as f:
		json_file = json.load(f)

	for data in json_file['data']:
		for paragraph in data['paragraphs']:
			r = []
			question_answers = []
			r.append(tokenize(paragraph['context']))
			for q in paragraph['qas']:
				answers = [a['text'] for a in q['answers']]
				if len(answers) > 0:
					ans_start = q['answers'][0]['answer_start']
					word_start_index = map_word_to_context(tokenize(paragraph['context']), ans_start)
					# if word_start_index is None or word_start_index.get(ans_start) is None:
					# 	continue
					if word_start_index is None:
						continue
					qa = [tokenize(q['question']),
						tokenize(answers[0]),
						[word_start_index, word_start_index + len(tokenize(answers[0]))]]

					question_answers.append(qa)
			if question_answers != None:
				r.append(question_answers)
				result.append(r)

	# Each element in result contains ['...context..', [['question 1', 'answer 1', 'start index'], ['question2', 'answer 2', 'start index']...]

	print('Serializing {} set object'.format(type))
	with open('{}_dataset.pkl'.format(type), 'wb') as f:
		pickle.dump(result, f)


def load_dataset(type):
	if type == 'train':
		path = 'train_dataset.pkl'
	else:
		path = 'test_dataset.pkl'
	with open(path, 'rb') as f:
		return pickle.load(f)

# generate_dataset('train')
# generate_dataset('test')
# print(load_dataset('train')[0][1])


def tokenize(sent):
	sent = sent.replace('"', "'").lower()
	return word_tokenize(sent)


class Vocabulary:

	def __init__(self, vocab_size, all_words):
		tokenizer = Tokenizer(num_words=vocab_size)
		tokenizer.fit_on_texts(all_words)
		self.vocab_size = len(tokenizer.word_index) + 1
		self.vocab = tokenizer.word_index
		self.reverse_vocab = tokenizer.index_word

vocab = Vocabulary(vocab_size=200000, all_words=[data[0] for data in load_dataset('train')])

with open('Vocabulary.pkl' ,'wb') as vf:
	pickle.dump(vocab, vf)

