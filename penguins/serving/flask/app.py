import tarfile
import tempfile
import numpy as np

from flask import Flask, request, jsonify
from pathlib import Path
from tensorflow import keras


MODEL_PATH = Path(__file__).parent


class Model:
    model = None

    def load(self):
        """
        Extracts the model package and loads the model in memory if it hasn't
        been loaded yet.
        """

        # We want to load the model only if it is not loaded yet.
        if not Model.model:
            # Before we load the model, we need to extract it in a
            # temporal directory.
            with tempfile.TemporaryDirectory() as directory:
                with tarfile.open(MODEL_PATH / "model.tar.gz") as tar:
                    tar.extractall(path=directory)

                Model.model = keras.models.load_model(directory)

    def predict(self, data):
        """
        Generates predictions for the supplied data.
        """

        self.load()
        return Model.model.predict(data)


app = Flask(__name__)
model = Model()


@app.route("/predict/", methods=["POST"])
def predict():
    data = request.data.decode("utf-8")

    data = np.array(data.split(",")).astype(np.float32)
    data = np.expand_dims(data, axis=0)

    predictions = model.predict(data=[data])

    prediction = int(np.argmax(predictions[0], axis=-1))
    confidence = float(predictions[0][prediction])

    return jsonify({"prediction": prediction, "confidence": confidence})
