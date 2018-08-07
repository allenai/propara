
Download glove embeddings file from https://nlp.stanford.edu/projects/glove/
and store it as data/glove.6B.100d.txt.gz

Command to train the ProLocal model
```
python propara/run.py train data/naacl18/prolocal/prolocal_params.json -s /tmp/pl1
```

Command for applying a pretrained ProLocal model to predict labels on test
```
python propara/run.py predict --output-file /tmp/prolocal.test.pred.json --predictor "prolocal-prediction" data/naacl18/prolocal/prolocal.model.tar.gz data/naacl18/prolocal/propara.run1.test.json
```

Command that takes the ProLocal model predictions and applies commonsense rules to complete the grids
```
to be added soon
```
