from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import mlflow
import os

with mlflow.start_run():

    #  Loading data
    digits = load_digits(as_frame=True)

    #  Splitting the data in the input and the resulting output
    x = digits.data
    y = digits.target

    # Splitting the data in test and training sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.50)

    # Feature Scaling training data to values between -1 and 1.
    # This makes it easier for the classifier to train
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # Constructing a classifier and using it to train a new model
    mlp = MLPClassifier(hidden_layer_sizes=(64, 128, 64), max_iter=1000, verbose=True)
    mlp.fit(x_train, y_train.values.ravel())

    print(mlp.get_params(deep=True))

    # Validating the model using test data
    predictions = mlp.predict(x_test)

    # Various logging
    print("----------------------------------------------------------------")
    print("Accuracy: {}".format(accuracy_score(y_test, predictions)))
    print("----------------------------------------------------------------")

    mlflow.set_tag("TestTag", "True")

    # Log an artifact (output file)
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    with open("outputs/test.txt", "w") as f:
        f.write("hello world!")
    mlflow.log_artifacts("outputs")
