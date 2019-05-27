import pickle
import numpy as np
from Model import build_model
import json
import params

with open('preprocessing.pkl', 'rb') as f:
    preprocessing = pickle.load(f)

with open('test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

model = build_model(preprocessing.tokenizer)
model.load_weights('Models/model_large.hdf5')

preprocessed_test_data = list()
for sample in test_data:
    if sample[3] < sample[4] < params.CONTEXT_LEN or sample[3] == -1:
        preprocessed_test_data.append(preprocessing.text_to_seq_sample(sample))

result = {}
i = 0
for sample in preprocessed_test_data:
    id_ = sample[6]
    pred = model.predict([sample[1].reshape(1, params.QUESTION_LEN), sample[0].reshape(1, params.CONTEXT_LEN)])

    p_start = int(np.argmax(pred[0]))

    p_end = int(np.argmax(pred[1]))
    is_noans = np.round(pred[2])
    result[id_] = preprocessing.target_to_words(sample[0].reshape(-1, ), p_start, p_end, is_noans)
    if (i + 1) % 250 == 0:
        print("{} samples done".format(i + 1))
    i += 1

for sample in test_data:
    if not (sample[3] < sample[4] < params.CONTEXT_LEN or sample[3] == -1):
        result[sample[6]] = ""

with open('results_large.json', 'w') as f:
    json.dump(result, f)
