from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics

from src import utils


if __name__ == "__main__":
    RANDOM_STATE = 64

    x, y = utils.read_image_classification_dataset(data_path="./out/train")
    print(f"loaded {len(y)} data points")
    utils.show_image_with_text(x[0], f"label {y[0]}")
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, test_size=0.15, shuffle=True, random_state=RANDOM_STATE)
    print(f"split data into: train {len(y_train)}, test {len(y_test)}")

    # classifier = MLPClassifier(solver="lbfgs", hidden_layer_sizes=(100, 100), random_state=RANDOM_STATE)
    # classifier = LinearSVC(random_state=RANDOM_STATE)  # 0.59833
    classifier = Perceptron(random_state=RANDOM_STATE)  # 0.58833
    print("defined classifier")
    classifier.fit(x_train, y_train)
    print("done fitting")
    score = classifier.score(x_test, y_test)
    print(score)
    predicted = classifier.predict(x_test)
    print(
        f"Classification report for classifier {classifier}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )
