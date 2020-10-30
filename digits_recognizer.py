from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import mlflow


with mlflow.start_run():

    mlflow.sklearn.autolog()

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

    # Validating the model using test data
    predictions = mlp.predict(x_test)

    mlflow.log_metric("Iterations", mlp.n_iter_)
    mlflow.log_metric("Accuracy", accuracy_score(y_test, predictions))
    for index in range(len(mlp.loss_curve_)):
        mlflow.log_metric("Loss", mlp.loss_curve_[index], step=index+1)
