import torchvision
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    # ---------------------------------------- load data ----------------------------------------
    trainData = torchvision.datasets.MNIST(
        root='MNIST/train',
        train=True,
        download=True
    )

    testData = torchvision.datasets.MNIST(
        root='MNIST/test',
        train=False,
        download=True
    )

    print("We have loaded our datasets")

    # ---------------------------------------- prepare data ----------------------------------------
    X_train = trainData.data.numpy()
    y_train = trainData.targets.numpy()

    X_test = testData.data.numpy()
    y_test = testData.targets.numpy()

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    X_train = X_train.reshape(len(X_train), 28 * 28)
    X_test = X_test.reshape(len(X_test), 28 * 28)

    # ---------------------------------------- model ----------------------------------------
    print("******** RANDOM FOREST MODEL ********")

    model = RandomForestClassifier(
        n_estimators=200,
        criterion="entropy",
        random_state=2,
        n_jobs=-1
    )

    # ---------------------------------------- training ----------------------------------------
    print("\n\n*****************\nTRAINING STARTING\n****************")
    model.fit(X_train, y_train)
    print("\n\n*****************\nTRAINING DONE\n****************")

    # save model
    joblib.dump(model, "random_forest_mnist_model.pkl")
    print("Random Forest model saved successfully")

    # ---------------------------------------- testing ----------------------------------------
    print("****************\nTESTING STARTING\n******************")
    y_pred = model.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_pred) * 100
    print("Test set accuracy = {:.2f} %".format(test_accuracy))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()