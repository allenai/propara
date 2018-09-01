Data and code related to our recent [EMNLP'18] (https://arxiv.org/abs/1808.10012) paper will be released here soon...
Reasoning about Actions and State Changes by Injecting Commonsense Knowledge, Niket Tandon, Bhavana Dalvi Mishra, Joel Grus, Wen-tau Yih, Antoine Bosselut, Peter Clark, EMNLP 2018


# ProPara
A repository of the state change prediction models used for evaluation in the __Tracking State Changes in Procedural Text: A Challenge Dataset and Models for Process Paragraph Comprehension__ paper accepted to NAACL'18. It contains
two models built using the PyTorch-based deep-learning NLP library, [AllenNLP](http://allennlp.org/).

 * ProLocal: A simple local model that takes a sentence and entity as input and predicts state changes happening to the entity. 
 * ProGlobal: A global model for state change prediction that takes entire paragraph and an entity as input and predicts the entity's state at every time-step in the paragraph. 
 
These models can be trained and evaluated as described below.

# Setup Instruction

1. Create the `propara` environment using Anaconda

  ```
  conda create -n propara python=3.6
  ```

2. Activate the environment

  ```
  source activate propara
  ```

3. Install the requirements in the environment: 

  ```
  pip install -r requirements.txt
  ```

4. Test installation

 ```
 pytest -v
 ```


# Download the dataset
You can download the dataset used in the NAACL'18 paper from 
  ```
   http://data.allenai.org/propara/
  ``` 

# Evaluate the models on ProPara dataset.
Example command to run eval script:
   ```
     python propara/eval/evalQA.py tests/fixtures/eval/para_id.test.txt tests/fixtures/eval/gold_labels.test.tsv tests/fixtures/eval/sample.model.test_predictions.tsv
   ```

If you find these models helpful in your work, please cite:
```
@inproceedings{proparNaacl2018,
     Author = { {Bhavana Dalvi, Lifu Huang}, Niket Tandon, Wen-tau Yih, Peter Clark},
     Booktitle = {NAACL},
     Title = {Tracking State Changes in Procedural Text: A Challenge Dataset and Models for Process Paragraph Comprehension},
     Year = {2018}
}

** Bhavana Dalvi and Lifu Huang contributed equally to this work.
```
