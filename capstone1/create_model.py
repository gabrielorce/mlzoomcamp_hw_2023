
import pandas as pd
#import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
import xgboost as xgb
import pickle

df = pd.read_csv('./data/hcvdat0.csv', index_col=0)
df.columns = df.columns.str.lower().str.replace(' ', '_')

xgb_params = {
    'eta': 0.4, 
    'max_depth': 3,
    'min_child_weight': 5,
    
    'objective': 'multi:softmax',   # this is a multiclass regression problem so we use multi:softmax instead of binary:logistic we saw in the course
    'num_class': 5,    # we are dealing with 5 possible classes as outcome
    'eval_metric': 'auc',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}

# show all categories and place them in dictionary
categs=df.category.unique()
categs


# let's create a dictionary to hold the male/female values. Also, replace all of the existing values with a numerical value
sex_definition=df.sex.unique()
sex_category={}

for c in sex_definition:
    if c=="m":
        sex_category[c]=0
    else:
        sex_category[c]=1
    df.replace(c, sex_category[c], inplace=True)


# let's create a dictionary to hold these values. Also, replace all of the existing values with a numerical 
patient_category={}

for c in categs:
    if c.split("=")[0]=="0s":
        patient_category[4]=c.split("=")[1]
        df.replace(c, 4, inplace=True)
    else:
        patient_category[int(c.split("=")[0])]=c.split("=")[1]
        df.replace(c, c.split("=")[0], inplace=True)
    
patient_category





# Do train/validation/test split with 60%/20%/20% distribution.
# Use the train_test_split function and set the random_state parameter to 1.
# reset indexes and delete target variable ("category") from original dataset

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

df_full_train = df_full_train.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# create target variable 
y_full_train = df_full_train.category.astype('int').values
y_train = df_train.category.astype('int').values   # astype('int') is for xgboost matrix
y_val = df_val.category.astype('int').values
y_test = df_test.category.astype('int').values

# delete the target variable from the datasets
del df_full_train['category']
del df_train['category']
del df_val['category']
del df_test['category']



def train (df_full_train, y_full_train):

    #Use DictVectorizer(sparse=True) to turn the dataframes into matrices.
    dicts_full_train = df_full_train.to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_full_train = dv.fit_transform(dicts_full_train)

    dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train,
                        feature_names=dv.get_feature_names_out())

    model = xgb.train(xgb_params, dfulltrain, num_boost_round=150)

    return dv, model

def predict(df, dv, model):
    dicts = df.to_dict(orient='records')
    X = dv.transform(dicts)
    dtest = xgb.DMatrix(X, feature_names=dv.get_feature_names_out())
    y_pred = model.predict(dtest)  # must predict probabilities for a multiclass problem
    return y_pred


# create model
    
dv, model = train(df_full_train, y_full_train)
y_pred = predict(df_test, dv, model)

print ("y_pred", y_pred)

outputfile= 'model_xgb.bin'
with open(outputfile, 'wb') as f_out:
    pickle.dump (model, f_out)

with open("dv.bin", 'wb') as f_out:
    pickle.dump (dv, f_out)

accuracy = accuracy_score(y_test, y_pred)
print(f"Done creating model named '{outputfile}' with Accuracy: %.2f%%" % (accuracy * 100.0))
