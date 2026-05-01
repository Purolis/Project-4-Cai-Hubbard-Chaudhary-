import os
import joblib
import numpy as np
from PIL import Image


def test_on_own_data():
    image_dir = "./img"
    model_name = "random_forest_mnist_model.pkl"

    model = joblib.load(model_name)

    total = 0
    correct = 0

    for filename in os.listdir(image_dir):
        if filename.endswith((".png" )):
            img_path = os.path.join(image_dir, filename)

            img = Image.open(img_path).convert("L")
            img = img.resize((28, 28))

            img_array = np.array(img)

            img_array = img_array / 255.0

            img_array = img_array.reshape(1, 28 * 28)

            prediction = model.predict(img_array)[0]

            print("File:", filename, "| Predicted Digit:", prediction)

            total += 1

            if str(prediction) == str(filename[0]):
                correct += 1

    print("Total images tested:", total)
    print("Correct predictions:", correct)

    if total > 0:
        accuracy = correct / total * 100
        print("Accuracy on written numbers:", accuracy, "%")
    else:
        print("No images found in the img folder.")


if __name__ == "__main__":
    test_on_own_data()