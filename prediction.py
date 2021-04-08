from sklearn import datasets
from joblib import load
import numpy as np
import json

#load the model

#my_model = load('svc_model.pkl')


iris_data = datasets.load_iris()
class_names = iris_data.target_names

def my_prediction(id):
    my_model = load('svc_model.pkl')
    dummy = np.array(id)
    dummyT = dummy.reshape(1,-1)
    dummy_str = dummy.tolist()
    r = dummy.shape
    t = dummyT.shape
    r_str = json.dumps(r)
    t_str = json.dumps(t)
    prediction = my_model.predict(dummyT)
    name = class_names[prediction]
    name = name.tolist()
    name_str = json.dumps(name)
    pred_str = prediction.tolist()
    pred_str = json.dumps(pred_str)
    dummy_str = json.dumps(dummy_str)
    str = ["The shape of the input is read as: ", r_str, "Reshaping the input array to get what the function is expecting: ",t_str, "The predicted value is: ",pred_str,"The flower associated with this prediction is: ", name_str, "The user observation was: ", dummy_str]
    return str
