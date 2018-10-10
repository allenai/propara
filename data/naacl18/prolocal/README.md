
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
python propara/utils/apply_inertia.py  --predictions data/naacl18/prolocal/output/propara.run1.test.pred.json  --full-grid data/naacl18/gold-full-grids.v3.tsv --output data/naacl18/prolocal/output/prolocal.naacl_cr.data_run1.completed.test.tsv
```

Command that takes the completed ProLocal model predictions and evaluates on NAACL'18 task:
```
python propara/eval/evalQA.py tests/fixtures/eval/para_id.test.txt tests/fixtures/eval/gold_labels.test.tsv data/naacl18/prolocal/output/prolocal.naacl_cr.data_run1.model_run2.test.tsv
```


