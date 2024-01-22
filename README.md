# Machine Learning with scikit-learn

## Getting started

### Prerequisites
The following should be installed before installing:
- git
- Python>=3.9 (https://www.python.org/downloads/)
- Poetry (https://python-poetry.org/docs/#installation)

### Installation
- Clone this repository
- Install required dependencies with running `poetry install --no-root` in repository root

## Workshop
Tip: Set a `RANDOM_STATE` before starting the following exercises. Most scikit-learn
operations have a `random_state`-argument. By providing it, you can recreate your results
if you wish to do so.
Tip: The documentation of scikit-learn is pretty good. Use it to find functionalities you
are looking for.

### 1. Load and split data
A utility function for reading the dataset is supplied in `utils.py`. However, this reads all
data. We should first split our dataset into two parts: a part for training, and a part for
testing. The training part is used to train our model, the testing part is used to determine
how well our model performs.

(Background: https://medium.com/@datasciencewizards/a-guide-to-data-splitting-in-machine-learning-49a959c95fa1)

Exercise: Split your dataset into a training and testing set.

### 2. Define and fit classifier
Next, we need a model or classifier to perform the classification. Scikit-learn provides a
number of models out of the box, for various applications. This is a good moment to look at
the documentation of scikit-learn.

Exercise: Define and train your first classifier. Keep it simple for now.

<details>
<summary>hint</summary>
If you are lost, you can start by looking at https://scikit-learn.org/stable/modules/linear_model.html#classification.
</details>
<details>
<summary>hint</summary>
Scikit-learn uses the term "fit" for training.
</details>

### 3. Predict with trained classifier
Now that we have our own trained model, let's use it to make some predictions! A utility
function for showing images with optional text is supplied in `utils.py`.

Exercise: Make some predictions with the classifier and visualize the image along with the
prediction.

### 4. Quantify performance
Visualising our predictions is interesting, but to really know how our model performs we
should quantify classifier performance. 

Question: on what data should we quantify our performance?
<details>
<summary>answer</summary>
We should use the test set, because the classifier is not trained on that data. If you use
the training set for determining the performance, you are at risk of missing overfitting.
</details>

Exercise: Quantify the performance of your classifier. (Tip: Remember the presentation from just
now.)

<details>
<summary>hint</summary>
If you are lost, you can start looking by at https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics.
</details>

### 5. Repeat
Now that we have trained our first classifier, let's try and improve! Real-world data science
problems need in iterative approach to get a good solution. First off, try and find a better
classifier.

Exercise: Improve the performance by evaluating different classifiers.

<details>
<summary>bonus</summary>
Look into tuning hyperparameters/behaviour of the classifier such as `learning_rate`,
`early_stopping`, `warm_start`. Try and find out what that means, but we are always close by
to help if you are stuck.
</details>
