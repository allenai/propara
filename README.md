# ProPara
A repository of the state change prediction models used for evaluation in the "Tracking State Changes in Procedural Text: A Challenge Dataset and Models for Process Paragraph Comprehension" paper accepted to NAACL'18. It contains
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
  sh scripts/install_requirements.sh
  ```

4. Install pytorch as per instructions on <http://pytorch.org/>. Commands as of Nov. 22, 2017:

  Linux/Mac (no CUDA): `conda install pytorch torchvision -c soumith`

  Linux   (with CUDA): `conda install pytorch torchvision cuda80 -c soumith`


6. Test installation

 ```
 pytest -v
 ```


# Download the dataset
You can download the dataset used in the NAACL'18 paper from 
  ```
   http://data.allenai.org/propara/
  ``` 

# Evaluate the models on ProPara dataset
   ```
     python eval/evalQA.py  data/para_ids.txt  data/gold_labels.test.tsv   <model-predictions-file-path>
   ```

If you find these models helpful in your work, please cite:
```
@inproceedings{proparNaacl2018,
     Author = {Bhavana Dalvi, Lifu Huang, Niket Tandon, Wen-tau Yih, Peter Clark},
     Booktitle = {NAACL},
     Title = {Tracking State Changes in Procedural Text: A Challenge Dataset and Models for Process Paragraph Comprehension},
     Year = {2018}
}
```
