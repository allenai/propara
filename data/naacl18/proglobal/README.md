Download the pretrained glove.6B.100d.txt embedding file from https://nlp.stanford.edu/projects/glove/ and put it under the folder: data/naacl18/proglobal/. Make sure this path to be the same as the pretrained_file in data/naacl18/proglobal/proglobal_params.json


Command to train the ProGlocal model, and save the output models in /tmp/pgl/

```
  python propara/runProGlobal.py train tests/fixtures/proglobal_params.json -s /tmp/pgl/
```

Command for applying a pretrained ProGlocal model to predict labels on test file: data/naacl/proglobal/all.chain.test.v3.recurssive.json
```
  python propara/runProGlobalPredictor.py predict data/naacl/proglobal/proglobal.model.tar.gz data/naacl/proglobal/all.chain.test.v3.recurssive.json --output-file data/naacl/proglobal/output/test.prediction.txt
```

Script for converting proglobal prediction output to evaluation format, with which you can directly run evalQA.py:
```
java -jar convertJson2Eval.jar data/naacl/proglobal/output/test.prediction.txt data/naacl/proglobal/output/test.prediction.eval.txt
```

The source function for convertJson2Eval.jar can be found in convertJson2Eval.java
