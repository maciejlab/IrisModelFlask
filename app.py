import pickle
from flask import Flask
from flask import request
from flask import jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Aplikacja do predykcji na zbiorze Iris"


@app.route('/app', methods=["GET"])
def get_prediction():

    sepal_length = float(request.args.get('sepal'))
    petal_length = float(request.args.get('pental'))

    features = [sepal_length, petal_length]

    class MyCustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "__main__":
                module = "program"
            return super().find_class(module, name)

    with open("perceptron_model.pickle", "rb") as f:
        unpickler = MyCustomUnpickler(f)
        file_pickle = unpickler.load()

    predicted_class = int(file_pickle.predict(features))

    return jsonify(features=features, predicted_class=predicted_class)


if __name__ == "__main__":
    app.run(host="0.0.0.0")