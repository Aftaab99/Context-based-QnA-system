from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize


def get_doc2vec(context):
	tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(context)]
	max_epochs = 100
	vec_size = 25
	alpha = 0.025

	model = Doc2Vec(size=vec_size,
					alpha=alpha,
					min_alpha=0.00025,
					min_count=1,
					dm=1)

	model.build_vocab(tagged_data)

	for epoch in range(max_epochs):
		model.train(tagged_data,
					total_examples=model.corpus_count,
					epochs=model.iter)
		# decrease the learning rate
		model.alpha -= 0.0002
		# fix the learning rate, no decay
		model.min_alpha = model.alpha

	return model, tagged_data
