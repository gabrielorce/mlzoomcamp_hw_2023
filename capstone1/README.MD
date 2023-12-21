# Mid-term project for MLZoomcamp 2023

## DATA

Retrieved from:
https://archive.ics.uci.edu/dataset/571/hcv+data  

It can also be retrieved from here:
https://github.com/uci-ml-repo/ucimlrepo  

but i chose to download it so i can add it to my source code (in case the dataset is erased in the future, for some reason)

The data set contains laboratory values of blood donors and Hepatitis C patients and demographic values like age.


## PURPOSE

The purpose of the model is to predict the category of the patient (blood donor, suspected blood donor, cirrhosis, Fibrosis, hepatitis) according to the features provided.

This dataset provides a number of features we can use to create the model:

- Age (in years) 
- Sex (male, female) 

  As well as the following lab results, mostly related to the the liver function: 

- ALB (albumin)
- ALP (alkaline phosphatase)
- ALT (alanine transaminase)
- AST (aspartate transaminase)
- BIL (serum bilirubi)
- CHE (serum cholinesterase)
- CHOL (cholesterol)
- CREA (creatinine blood)
- GGT (gamma-glutamyl transferas)
- PROT (total protein)

## REQUIREMENTS
Be sure to install xgboost:
```
sudo pip install xgboost
```


## DETAILS
Since this is a multi-class classification problem, we will use "multi:softprob" as the XGBoost loss function.

The create_model.py script allows you to create the xgboost model, execute it and you will crewte a file named "model_xgb.bin". You can modify the parameters at the top of the file to get a different model.


You can execute the predictor in one of two ways:
Run te predict_category.py script, which runs flask at http://localhost:9696/predict
You must post values as a JSON. You can post information on one pantient with prefilled information using the script called patient_request.py

The other way is to build the docker image using the steps below. Once run, you will also be able to use the predictor via POST requests to  http://localhost:9696/predict

DOCKER STEPS:
1) BUILD IT
```
docker build -t predict_category .
```

2) RUN IT
```
docker run -it --rm -p 9696:9696 predict_category
```
