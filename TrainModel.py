from keras.layers import LSTM, Input, Dense, RepeatVector, Embedding
from Preprocessing import Preprocessing
import pickle

CONTEXT_LEN = 50
QUESTION_LEN = 15
ANSWER_LEN = 15
EMBEDDING_SIZE = 300

preprocessing = Preprocessing(size=15000)

# Loading the train data
with open('train_data.pkl', 'rb') as f:
	train_data = pickle.load(f)

texts = []
for item in train_data:
	for i in item:
		texts.append(i)

preprocessing.create_vocabulary(texts)
print('Vocabulary size: {}'.format(preprocessing.vocab_size))

preprocessed_train_data = list()
for sample in train_data:
	preprocessed_train_data.append(preprocessing.text_to_seq_sample(sample))

with open('preprocessing.pkl', 'wb') as f:
	pickle.dump(preprocessing, f)

print('Done with processing..')
print(preprocessing.sequence_to_text(preprocessed_train_data[65][0]))
