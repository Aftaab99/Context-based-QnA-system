from keras.models import load_model
import pickle
import numpy as np
from Model import build_model
import json
from random import shuffle

with open('preprocessing.pkl', 'rb') as f:
    preprocessing = pickle.load(f)

with open('test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

model = build_model(preprocessing.tokenizer)
model.load_weights('Models/weights-improvement-02-10.59.hdf5')

preprocessed_test_data = list()
for sample in test_data:
    if sample[3] < sample[4] < 325 or sample[3] == -1:
        preprocessed_test_data.append(preprocessing.text_to_seq_sample(sample))

result = {}
i = 0
for sample in preprocessed_test_data:
    id_ = sample[6]
    pred = model.predict([sample[1].reshape(1, 22), sample[0].reshape(1, 325)])

    p_start = int(np.argmax(pred[0]))

    p_end = int(np.argmax(pred[1]))
    is_noans = np.round(pred[2])
    result[id_] = preprocessing.target_to_words(sample[0].reshape(-1, ), p_start, p_end, is_noans)
    if (i + 1) % 250 == 0:
        print("{} samples done".format(i + 1))
    i += 1

for sample in test_data:
    if not (sample[3] < sample[4] < 325 or sample[3] == -1):
        result[sample[6]] = ""

with open('results.json', 'w') as f:
    json.dump(result, f)
