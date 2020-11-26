
# load 
from pandas import read_csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from autokeras import StructuredDataClassifier
import tensorflow as tf
import keras
import numpy as np
from numpy import asarray

# load dataset
data_to_predict = pd.read_pickle("../data/davidsSet.pickle")
data_to_predict['duration'] = [0.0 if isinstance(x,str) else x for x in data_to_predict['duration']]
#load Modell
loaded = keras.models.load_model("model_AI4Sec")

#No nessecary .. ModelSummary/Architecture
loaded.summary()

yhat = loaded.predict(data_to_predict)
print('Predicted: %.3f' % yhat[0])

#Not nessecary .. DiplayPredicted Values
new = pd.DataFrame(yhat)
print(new.describe())



