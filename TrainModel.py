from Preprocessing import Preprocessing
from Model import build_model, keras_focal_loss
import pickle
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.utils.generic_utils import get_custom_objects

get_custom_objects().update({"keras_focal_loss": keras_focal_loss})

CONTEXT_LEN = 50
QUESTION_LEN = 15
ANSWER_LEN = 15
EMBEDDING_SIZE = 300

preprocessing = Preprocessing(size=90000)

# Loading the train data
with open('train_data.pkl', 'rb') as f:
	train_data = pickle.load(f)

texts = []
for item in train_data:
	for i in item[0:3]:
		texts.append(i)

preprocessing.create_vocabulary(texts)
print('Vocabulary size: {}'.format(preprocessing.vocab_size))

preprocessed_train_data = list()
for sample in train_data:
	preprocessed_train_data.append(preprocessing.text_to_seq_sample(sample))

with open('preprocessing.pkl', 'wb') as f:
	pickle.dump(preprocessing, f)

print('Done with processing..')
c = preprocessed_train_data[65][0]
print(preprocessing.sequence_to_text(c))
x = preprocessed_train_data[65][3]

# model = build_model(preprocessing.tokenizer)
model = load_model('Models/weights-improvement-06-173.36.hdf5')

filepath = "Models/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# context0 = preprocessed_train_data[0][0]
# question0 = preprocessed_train_data[0][1]
# answer_start0 = preprocessed_train_data[0][3]
# answer_end0 = preprocessed_train_data[0][4]
contexts = np.array([c[0] for c in preprocessed_train_data]).reshape(-1, 1000)
questions = np.array([q[1] for q in preprocessed_train_data]).reshape(-1, 50)
target = np.array([x[3] for x in preprocessed_train_data]).reshape(-1, 1000)

model.fit([questions, contexts], target, epochs=10,
		validation_split=0.15, callbacks=callbacks_list, batch_size=64, verbose=1)
# model.fit([question0, context0], np.array([answer_start0, answer_end0]).reshape(-1, 2))
# model.save('Models/test-model.hdf5')
