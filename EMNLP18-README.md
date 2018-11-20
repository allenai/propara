Initialize the environment created as per the file README.md in the root directory.
```
    source activate propara
    export PYTHONPATH=.
```

Command to train a ProStruct model

```
    python propara/run.py train data/emnlp18/prostruct_params_local.json -s /tmp/prostruct1
```

Command for applying a pretrained ProStruct model to predict labels on test

```
    python propara/run.py predict --output-file /tmp/prostruct1/test.pred.json --predictor "prostruct_prediction" /tmp/prostruct1/model.tar.gz data/emnlp18/grids.v1.test.json
```

Command that takes the ProStruct model predictions in json format and converts them to TSV format needed for the evaluator

```
    python propara/utils/prostruct_predicted_json_to_tsv_grid.py /tmp/pl1/predictions/pred.test.json /tmp/pl1/predictions/pred.test.tsv
```

To evaluate your model's predictions on the ProPara task (EMNLP'18),
please Download the evaluator code from a separate leaderboard repository: (https://github.com/allenai/aristo-leaderboard/tree/master/propara)


ProPara leaderboard is now live at: (https://leaderboard.allenai.org/propara)