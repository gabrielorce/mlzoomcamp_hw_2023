# this s used in the dockerfile. THE ORIGINAL DOKERFILE ALREADY CONTAINS THE MODEL (model2.bin)

# can POST info to localhost:9696 (or 0.0.0.0:9696)
# TO EXECUTE WITH FLASK: 
# just execute script
# TO EXECUTE with gunicorn:
# pipenv install flask gunicorn
# gunicorn --bind 0.0.0.0:9696  q5_predict_client:app

import pickle
from flask import Flask
from flask import request
from flask import jsonify

app = Flask('credit-card')


# load dicvectorizer and model
with open('dv.bin', 'rb') as f_in:
    dv = pickle.load(f_in)

with open('model2.bin', 'rb') as f_in:
    model = pickle.load(f_in)
    

# set up route and method
@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()

    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]
    get_card = y_pred >= 0.5

    result = {
        'get_card_probability': float(y_pred),
        'get_card': bool(get_card)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)