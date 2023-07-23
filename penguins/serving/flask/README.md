# Serving the model using Flask

A Flask application that serves a multi-class classification TensorFlow model to determine the species of a penguin.

This example is part of the [Machine Learning School](https://www.ml.school) program.

To run the application, start by creating a virtual environment and installing the `requirements.txt` file:

```bash
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

The following command will run the Flask server on `locahost` port `4242`:

```bash
$ export FLASK_ENV=development && flask run --host=0.0.0.0 --port=4242
```

Run the following command to test the server by predicting the species of a penguin:

```bash
$ curl --location --request POST 'http://localhost:4242/predict' \
--header 'Content-Type: text/plain' \
--data-raw '0.6569590202313976, -1.0813829646495108, 1.2097102831892812, 0.9226343641317372, 1.0, 0.0, 0.0'
```

You should get back the following JSON response:

```json
{
    "confidence": 0.8113499283790588,
    "prediction": 2
}
```
