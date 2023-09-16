# Interpretable Neural Networks
Work on Interpretable Neural Networks based on IAIA_BL thesis for the summer of 2023 research project - with Professor Luk Wayne.
Contains code for training and testing an interpretable model that consist of a BlackBox NN, followed by prototype layer that 
calculates distance between prototypes and convolutional features. The final fully connected layer gives the output logits for inference.

Note:- Please find the final versions of the scripts on the '*final-master-orphan*' branch

## Files
All files used for creating and testing models. Main.py to train models and accuracy.py to test them. Checkout
respective bash scripts for running these files. The ProtoPNet architecture can be found in model.py.
##### Note:- Please find the final versions of the scripts on the '*final-master-orphan*' branch

## Data
Database of vegetables with 3 classes to imitate breast lesion database (which was not available at the time).
Has various subfolders for training, testing and vaildation etc.

## ModelStore
History of models created, 
