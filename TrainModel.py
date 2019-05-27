from Preprocessing import Preprocessing
from Model import build_model
import pickle
import numpy as np
from keras.callbacks import ModelCheckpoint
import params


preprocessing = Preprocessing(size=params.VOCAB_SIZE)

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
    if sample[4] < params.CONTEXT_LEN or sample[3] == -1:  # Only considering contexts smaller than 325 words
        preprocessed_train_data.append(preprocessing.text_to_seq_sample(sample))

with open('preprocessing.pkl', 'wb') as f:
    pickle.dump(preprocessing, f)

print('Done with processing..')

model = build_model(preprocessing.tokenizer)

filepath = "Models/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

contexts = np.array([c[0] for c in preprocessed_train_data]).reshape(-1, params.CONTEXT_LEN)
questions = np.array([q[1] for q in preprocessed_train_data]).reshape(-1, params.QUESTION_LEN)
target_start = np.array([x[3] for x in preprocessed_train_data]).reshape(-1, params.CONTEXT_LEN)
target_end = np.array([x[4] for x in preprocessed_train_data]).reshape(-1, params.CONTEXT_LEN)
is_no_ans = np.array([x[5] for x in preprocessed_train_data]).reshape(-1, 2)
model.fit([questions, contexts], [target_start, target_end, is_no_ans], epochs=10,
          validation_split=0.05, callbacks=callbacks_list, batch_size=64, verbose=1)
model.save('Models/model_large.hdf5')
