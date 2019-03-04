import json
import pickle
from nltk.tokenize import word_tokenize, sent_tokenize


def load_dataset(dataset_type):
	if dataset_type == 'train':
		path = 'Dataset/train-v2.0.json'
	else:
		path = 'Dataset/dev-v2.0.json'

	with open(path, 'r') as f:
		json_file = json.load(f)
	current_article = 1
	res = []
	q_count = 0
	article_q = 0
	for data in json_file['data']:

		for paragraph in data['paragraphs']:
			context = paragraph['context']
			sentences = sent_tokenize(context)
			if '' in sentences:
				sentences.remove('')

			for qa in paragraph['qas']:
				if qa['is_impossible']:
					continue

				question = qa['question']
				ax = [a['text'] for a in qa['answers']]
				if not len(ax) > 0:
					continue

				answer = ax[0]
				answer_start = [a['answer_start'] for a in qa['answers']]
				answer_end = [a['answer_start'] + len(a['text'].split(' ')) for a in qa['answers']]

				if q_count % 250 == 0 and q_count != 0:
					print('{} questions done'.format(q_count))

				q_count += 1
				article_q += 1
				for a, a_start, a_end in zip(ax, answer_start, answer_end):
					r = [context, question, a, a_start, a_end]
					res.append(r)

		print('Article {}, no. of questions={}'.format(current_article, article_q))
		current_article += 1
		article_q = 0

	return res


train = load_dataset('train')
with open('train_data.pkl', 'wb') as f:
	pickle.dump(train, f)

test = load_dataset('test')
with open('test_data.pkl', 'wb') as f:
	pickle.dump(test, f)
