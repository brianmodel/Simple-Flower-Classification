# Simple Flower Classification

This project mainly serves as a blueprint for future, more complicated models in Tensorflow. 

The FlowerClassification.ipynb jupyter notebook file can be used to see the effects that most of the operations have onto the data directly.

Finally, make sure to modify the optimizer/cost function as well as the accuracy/metrics function for different datasets as it may not be the same as it is here (Gradient Descent Optimizer and a simple number correct/total predictions cost).

# Modifying nextbatch.py
The nextbatch.py file can be used when doing mini-batch gradient descent for creating the mini batches that will be fed into the input layer. When using this file, be aware of the following:
- The data must be randomly organized prior to being fed into this function. A simple way to do this is by doing a train_test_split from the sklearn.model_selected library.
- If this function is used for other projects, make sure the data type of your X matrix and y vector are correct for this function (i.e. know whether your data is stored as a numpy array or a pandas DataFrame or Series)
