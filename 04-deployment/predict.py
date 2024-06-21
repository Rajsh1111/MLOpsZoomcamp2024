import pickle

from flask import Flask, request, jsonify

from sklearn.feature_extraction.text import DictVectorizer
from sklearn.linear_model import LinearRegression

# Load the vectorizer
with open('vectorizer.pkl', 'rb') as f:
    dv= pickle.load(f)

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


def prepare_features(ride):
    features = {}
    features['PULocationID'] = ride['PULocationID']
    features['DOLocationID'] = ride['DOLocationID']
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(features):
    X = dictvectorizer.transform(features)
    preds = model.predict(X)
    return float(preds[0])

app = Flask('duration-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration': pred
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)