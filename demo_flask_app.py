from flask import Flask, request, render_template, jsonify
import pickle
from Model import build_model
import params
import numpy as np
import tensorflow as tf
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        context = request.form.get("context")
        question = request.form.get("question")
        global graph
        with graph.as_default():
            context_pre = preprocessing.text_to_seq(context, params.CONTEXT_LEN)
            question_pre = preprocessing.text_to_seq(question, params.QUESTION_LEN)

            pred = model.predict([question_pre.reshape(
                1, params.QUESTION_LEN), context_pre.reshape(1, params.CONTEXT_LEN)])
            p_start = int(np.argmax(pred[0]))

            p_end = int(np.argmax(pred[1]))
            is_noans = np.round(pred[2])
        print('done from server')
        r = {'ans': preprocessing.target_to_words(
            context_pre.reshape(-1, ), p_start, p_end, is_noans)}
        print(r)
        return jsonify(r)

with open('preprocessing.pkl', 'rb') as f:
    preprocessing = pickle.load(f)

model = build_model(preprocessing.tokenizer)
model.load_weights('Models/model_large.hdf5')
graph = tf.get_default_graph()

app.run(debug=True)
