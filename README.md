# EMNLP 2018 Update
Data and code related to our recent [EMNLP'18 paper] (https://arxiv.org/abs/1808.10012) is released on 31st Oct 2018.

**Code contributors: Niket Tandon, Bhavana Dalvi Mishra, Joel Grus

Detailed instructions to train your own ProStruct model can be found in: EMNLP18-README.md

Evaluate your model's predictions on the ProPara task (EMNLP'18):
```
    Download the evaluator code from a separate leaderboard repository:
    https://github.com/allenai/aristo-leaderboard/tree/master/propara
```

ProPara leaderboard is now live at:
```
https://leaderboard.allenai.org/propara/submissions/get-started
```

# ProPara
The ProPara dataset is designed to train and test comprehension of simple paragraphs describing processes, e.g., photosynthesis. We treat the comprehension task as that of predicting, tracking, and answering questions about how entities change during the process.

This repository contains code following three neural models developed at Allen Institute for Artificial Intelligence.
These models are built using the PyTorch-based deep-learning NLP library, [AllenNLP](http://allennlp.org/).

 * ProLocal: A simple local model that takes a sentence and entity as input and predicts state changes happening to the entity. 
 * ProGlobal: A global model for state change prediction that takes entire paragraph and an entity as input and predicts the entity's state at every time-step in the paragraph. 
 * ProStruct: A global model for state change prediction that incorporates commonsense to output most sensible predictions for the entire paragraph.

ProLocal and Proglobal are described in our NAACL'18 paper.

  ```
    Reasoning about Actions and State Changes by Injecting Commonsense Knowledge, Bhavana Dalvi Mishra, Lifu Huang, Niket Tandon, Wen-tau Yih, Peter Clark, NAACL 2018
  ```
  ** Bhavana Dalvi Mishra and Lifu Huang contributed equally to this work.


ProStruct model is described in our EMNLP'18 paper:
   ```
    Reasoning about Actions and State Changes by Injecting Commonsense Knowledge, Niket Tandon, Bhavana Dalvi Mishra, Joel Grus, Wen-tau Yih, Antoine Bosselut, Peter Clark, EMNLP 2018
   ```
   ** Niket Tandon and Bhavana Dalvi Mishra contributed equally to this work.

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
You can download the ProPara dataset from
  ```
   http://data.allenai.org/propara/
  ``` 

# Train your own models
Detailed instructions are given in the following READMEs:
 * ProLocal: data/naacl18/prolocal/README.md
 * ProGlobal: data/naacl18/proglobal/README.md
 * ProStruct: EMNLP18-README.md

If you find these models helpful in your work, please cite:
```
@article{proparNaacl2018,
     Title = {Tracking State Changes in Procedural Text: A Challenge Dataset and Models for Process Paragraph Comprehension},
     Author = {Bhavana Dalvi and Lifu Huang and Niket Tandon and Wen-tau Yih and Peter Clark},
     journal = {NAACL},
     Year = {2018}
}

@article{prostructEmnlp2018,
  title={Reasoning about Actions and State Changes by Injecting Commonsense Knowledge},
  author={Niket Tandon and Bhavana Dalvi Mishra and Joel Grus and Wen-tau Yih and Antoine Bosselut and Peter Clark},
  journal={EMNLP},
  year={2018},
}
```
