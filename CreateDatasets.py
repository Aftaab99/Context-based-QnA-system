import json
from TrainDoc2VecModel import get_doc2vec
import pickle
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize, sent_tokenize


def load_dataset(dataset_type, max_questions=2000):
	global infersent

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
			model_doc2vec, tagged_context = get_doc2vec(sentences)

			for qa in paragraph['qas']:
				if qa['is_impossible']:
					continue

				question = qa['question']
				ax = [a['text'] for a in qa['answers']]
				if not len(ax) > 0:
					continue

				answer = ax[0]
				new_sentence = word_tokenize(question.lower())

				highest_tags = model_doc2vec.docvecs.most_similar(positive=[model_doc2vec.infer_vector(new_sentence)],
																 topn=5)

				focused_context = ''
				prev_sent = ''
				for ht in highest_tags:
					tag_index = ht[1]
					if prev_sent != sentences[int(tag_index)]:
						focused_context = focused_context+' '+sentences[int(tag_index)]
						prev_sent = sentences[int(tag_index)]

				focused_context=focused_context.strip()
				if max_questions is not None and q_count == max_questions:
					return res
				if q_count % 250 == 0 and q_count != 0:
					print('{} questions done'.format(q_count))

				q_count += 1
				article_q += 1
				r = [focused_context, question, answer]
				res.append(r)

		print('Article {}, no. of questions={}'.format(current_article, article_q))
		current_article += 1
		article_q = 0

	return res


x = load_dataset('train', max_questions=2000)
train, validation = train_test_split(x, test_size=0.15)
with open('train_data.pkl', 'wb') as f:
	pickle.dump(train, f)
with open('validation_data.pkl', 'wb') as f:
	pickle.dump(validation, f)

test = load_dataset('test', max_questions=2000)
with open('test_data.pkl', 'wb') as f:
	pickle.dump(test, f)
