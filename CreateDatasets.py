import json
import pickle
from nltk.tokenize import sent_tokenize


def load_dataset(dataset_type):
    if dataset_type == 'train':
        path = 'Datasets/train-v2.0.json'
    else:
        path = 'Datasets/dev-v2.0.json'

    with open(path, 'r') as f:
        json_file = json.load(f)
    current_article = 1
    res = []
    q_count = 0
    article_q = 0
    is_imp = 0
    for data in json_file['data']:

        for paragraph in data['paragraphs']:
            context = paragraph['context']
            sentences = sent_tokenize(context)
            if '' in sentences:
                sentences.remove('')

            for qa in paragraph['qas']:

                question = qa['question']
                ax = [a['text'] for a in qa['answers']]
                if qa['is_impossible']:
                    is_imp += 1
                    if dataset_type == 'test':
                        print('Added qa impossible id')
                        res.append([context, question, '<No answer>', -1, -1, [0, 1], qa['id']])
                    else:
                        res.append([context, question, '<No answer>', -1, -1, [0, 1]])
                else:
                    answer_start = [a['answer_start'] for a in qa['answers']]
                    answer_end = [a['answer_start'] + len(a['text'].split(' ')) for a in qa['answers']]
                    for a, a_start, a_end in zip(ax, answer_start, answer_end):
                        if dataset_type == 'test':
                            print('Added qa id')
                            r = [context, question, a, a_start, a_end, [1, 0], qa['id']]
                        else:
                            r = [context, question, a, a_start, a_end, [1, 0]]
                        res.append(r)

                if q_count % 250 == 0 and q_count != 0:
                    print('{} questions done'.format(q_count))

                q_count += 1
                article_q += 1

        print('Article {}, no. of questions={}'.format(current_article, article_q))
        current_article += 1
        article_q = 0
    print('{} impossible questions'.format(is_imp))
    return res


train = load_dataset('train')
with open('train_data.pkl', 'wb') as f:
    pickle.dump(train, f)

test = load_dataset('test')
with open('test_data.pkl', 'wb') as f:
    pickle.dump(test, f)
