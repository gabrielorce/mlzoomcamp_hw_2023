# this is used in the dockerfile.

# can POST info to localhost:9696 (or 0.0.0.0:9696)
# TO EXECUTE WITH FLASK: 
# just execute script
# TO EXECUTE with gunicorn:
# pipenv install flask gunicorn
# gunicorn --bind 0.0.0.0:9696  predict_category:app

import pickle
from flask import Flask
from flask import request
from flask import jsonify

app = Flask('predict_category')


# load dictvectorizer and model
with open('dv.bin', 'rb') as f_in:
    dv = pickle.load(f_in)

with open('model_xgb.bin', 'rb') as f_in:
    model = pickle.load(f_in)
    

# set up route and method
@app.route('/predict', methods=['POST'])
def predict():
    patient = request.get_json()
    print (patient)
    X = dv.transform([patient])
    print ("X:",X)
    y_pred = model.predict(X)[0, 1]

    result = {
        'patient category': y_pred,
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)