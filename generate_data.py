import json

def generate_dataset(type):
	if type == 'train':
		path = 'Dataset/train-v2.0.json'
	else:
		path = 'Dataset/dev-v2.0.json'

	result=[]

	with open(path, 'r') as f:
		json_file = json.load(f)

	for data in json_file['data']:
		for paragraph in data['paragraphs']:
			r=[]
			question_answers=[]
			r.append(paragraph['context'])
			for q in paragraph['qas']:
				answers = [a['text'] for a in q['answers']]
				if len(answers)>0:
					qa=[q['question'], answers[0]]
					question_answers.append(qa)
			if question_answers!=None:
				r.append(question_answers)
				result.append(r)
	# Each element in result contains ['...context..', [['question 1', 'answer 1], ['question2', 'answer 2']...]
	print(result[4][0])

generate_dataset('dev')