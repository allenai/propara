Download glove embeddings file from https://nlp.stanford.edu/projects/glove/ and store it as data/glove.6B.100d.txt.gz

Initialize the environment created as per the main README in the project
```
    source activate processes
    export PYTHONPATH=.
```

Command to train a ProStruct model

```
    python processes/run.py train data/emnlp18/prostruct_params_local.json -s /tmp/prostruct1
```

Command for applying a pretrained ProStruct model to predict labels on test

```
    python processes/run.py predict --output-file /tmp/prostruct1/test.pred.json --predictor "prostruct_prediction" /tmp/prostruct1/model.tar.gz data/emnlp18/grids.v1.test.json
```

Command that takes the ProStruct model predictions in json format and converts them to TSV format needed for the evaluator

```
    python processes/utils/prostruct_predicted_json_to_tsv_grid.py /tmp/pl1/predictions/pred.test.json /tmp/pl1/predictions/pred.test.tsv
```

Evaluate a model's predictions on the EMNLP'18 task:

```
    Download the evaluator code from AI2 leaderboard website:
    “code link with the instructions to run the evaluation will be available soon”

```