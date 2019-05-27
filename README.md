# Context based Question and Answering system

A context based question answering system using a variation of the model described in
[this paper](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1184/reports/6904508.pdf)

Unlike the paper, this implementation is trained on SQUAD v2.0, which is a much harder dataset, as it has many questions which are impossible to answer based on information given in the context. 


## Setting up

    conda env create --name qa_env -f qnaenv.yaml
    
To download the pretrained glove word embeddings, dataset and pretrained model run,
    
    python DownloadData.py 

## Training
As this isn't a relatively large model, you can train it on CPU, with minimum 8 GB system memory. On a Intel i5, each epoch takes roughly 35min and you'll probably need to train it for at least 20 epochs to see good results.
The authors have used 200 GRU units, so that maybe helpful, incase yoo have the computational means to train the model. 

## Testing
Generate a predictions file, `results.json` using

    python Test.py

Evaluate the model using the model using the official evaluation script you can run,

    python evaluate.py --data_file Datasets/dev-v2.0.json --pred_file results.json


## Pre-trained model scores

1) Small model(50 GRU units)

        {
          "exact": 22.311126084393162,
          "f1": 23.42393885183988,
          "total": 11873,
          "HasAns_exact": 0.4892037786774629,
          "HasAns_f1": 2.7180205782548734,
          "HasAns_total": 5928,
          "NoAns_exact": 44.07064760302775,
          "NoAns_f1": 44.07064760302775,
          "NoAns_total": 5945
        }

2) Large model(200 GRU units, 10 epochs)

        {
          "exact": 25.38532805525141,
          "f1": 26.50248099500217,
          "total": 11873,
          "HasAns_exact": 0.4048582995951417,
          "HasAns_f1": 2.6423678902936114,
          "HasAns_total": 5928,
          "NoAns_exact": 50.29436501261564,
          "NoAns_f1": 50.29436501261564,
          "NoAns_total": 5945
        }
