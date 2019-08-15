# EMNLP 2019 Update

Depedency graph dataset is available to download at:
[link](https://docs.google.com/spreadsheets/d/1UOqqJGstexgtWxMiNU090ALo9Dbd2Z7i4t441_FRk44/edit?usp=sharing)

Contact: {bhavanad,nikett,peterc}@allenai.org	

If you find this dataset helpful in your work, please cite:
```
@article{xpademnlp2019,
     Title = {Everything Happens for a Reason: Discovering the Purpose of Actions in Procedural Text},
     Author = {Bhavana Dalvi, Niket Tandon, Antoine Bosselut, Wen-tau Yih and Peter Clark},
     journal = {EMNLP},
     Year = {2019}
}```


The XPAD Dependency Graph Dataset	
==================================

This dataset supplements the ProPara dataset described in Dalvi et al., Tracking state changes in procedural text: A challenge dataset and models for process paragraph comprehension (NAACL'18). ProPara is available at [link](https://docs.google.com/spreadsheets/d/1x5Ct8EmQs2hVKOYX7b2nS0AOoQi4iM7H9d9isXRDwgM).


This supplementary dataset is called the dependency graph dataset, listing dependency relations between steps in the ProPara dataset. Fields are as follows:	
	
* PID:	The ID of the process paragraph
* StepNo:	The step number of each sentence in the paragraph
* enables:	set to "enables", if StepNo is annotated as enabling a future step
* EnabledStepNo:	The future step number enabled by StepNo
* Step: 	The text of StepNo
* enables:	repeat of the earlier column, for legibility
* EnabledStep:	The text of EnabledStepNo, the future step enabled by Step
* State Change:	The state change in Step, drawn from the original ProPara Turked annotations. This state change allows EnabledStep to proceed.
* Entity:	The entity involved in State Change
* FromLoc:	For a Move state change, where the move is from
* ToLoc:	For a Move state change, where the move is to
* Type: 	Justification for the enables edge, namely either Entity is "mentioned" or "changed" in the EnabledStep.
	
       
For legibility, steps with no recognized dependencies (due to the limited expressive power of the ProPara state change vocabulary) are also listed, but without "enables" edges.	
	
       
Given the ProPara annotations for State Change, and this dataset annotating dependencies between steps, the learning task is to predict the enables edges and the justification (State Change, Entity, FromLoc, ToLoc) between steps in the test set.	

Each dependency contains six elements: enabling-step-id, enabled-step-id, entity, state-change-type, from-location, to-location. Predictions are scored by counting each element correctly predicted, to compute Precision/Recall/F1 for each paragraph. The overall score is these metrics averaged over all paragraphs in the test partition of the dataset.	
