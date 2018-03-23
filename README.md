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
  sh scripts/install_requirements.sh
  ```

4. Install pytorch version torch (0.3.1) as per instructions on <http://pytorch.org/>. 
   Commands as of Mar. 20, 2018:

  Mac (no CUDA): `pip3 install http://download.pytorch.org/whl/torch-0.3.1-cp36-cp36m-macosx_10_7_x86_64.whl`

  Linux   (no CUDA): `pip3 install http://download.pytorch.org/whl/cpu/torch-0.3.1-cp36-cp36m-linux_x86_64.whl `


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
     Author = { {Bhavana Dalvi, Lifu Huang}, Niket Tandon, Wen-tau Yih, Peter Clark},
     Booktitle = {NAACL},
     Title = {Tracking State Changes in Procedural Text: A Challenge Dataset and Models for Process Paragraph Comprehension},
     Year = {2018}
}

** Bhavana Dalvi and Lifu Huang contributed equally to this work.
```
