# Origin of the model and dictVect:
# PREFIX=https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/master/cohorts/2023/05-deployment/homework
# wget $PREFIX/model1.bin
# wget $PREFIX/dv.bin

# check their checksum:
# md5sum model1.bin dv.bin
# 8ebfdf20010cfc7f545c43e3b52fc8a1  model1.bin
# 924b496a89148b422c74a62dbc92a4fb  dv.bin


import pickle


with open('dv.bin', 'rb') as f_in:
    dv = pickle.load(f_in)

with open('model1.bin', 'rb') as f_in:
    model = pickle.load(f_in)
    

client = {"job": "retired", "duration": 445, "poutcome": "success"}


X = dv.transform([client])
y_pred = model.predict_proba(X)[0, 1]

print(y_pred)