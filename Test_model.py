from keras.models import load_model
import pickle
import numpy as np
from Model import keras_focal_loss
from keras.utils.generic_utils import get_custom_objects

get_custom_objects().update({"keras_focal_loss": keras_focal_loss})

with open('preprocessing.pkl', 'rb') as f:
	preprocessing = pickle.load(f)

with open('train_data.pkl', 'rb') as f:
	test_data = pickle.load(f)

model = load_model('Models/weights-improvement-06-173.36.hdf5')
preprocessed_test_data = list()
for sample in test_data:
	preprocessed_test_data.append(preprocessing.text_to_seq_sample(sample))

contexts = np.array([c[0] for c in preprocessed_test_data]).reshape(-1, 1000)
questions = np.array([q[1] for q in preprocessed_test_data]).reshape(-1, 50)
target = np.array([x[3] for x in preprocessed_test_data]).reshape(-1, 1000)
context0 = preprocessed_test_data[65][0]
question0 = preprocessed_test_data[65][1]
target0 = preprocessed_test_data[65][3]
x = model.predict([question0, context0])
print(x)
print('Correct ans: %s, Predicted ans: %s' % (preprocessing.sequence_to_text(preprocessed_test_data[65][2]),
											  preprocessing.sequence_to_text(np.round(x))))
# print(target.shape)
# acc = model.evaluate([questions, contexts], target)
# print('Test loss={}'.format(acc[0]))
