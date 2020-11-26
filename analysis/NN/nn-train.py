# load the sonar dataset
from pandas import read_csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from autokeras import StructuredDataClassifier

# load dataset
#url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
dataframe = pd.read_pickle("../data/unsw-small.pickle")
dataframe


# split into input and output elements
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
print("shape of the input")
print(X.shape, y.shape)
X = X.astype('float32')
y = LabelEncoder().fit_transform(y)
# separate into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("shape of the splitting Data X_train, X_test, y_train, y_test")
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# define the search space
search = StructuredDataClassifier(max_trials=15)

# perform the search
search.fit(x=X_train, y=y_train, verbose=1)

# evaluate the model
loss, acc = search.evaluate(X_test, y_test, verbose=0)
print('Modell Accuracy')
print('Accuracy: %.3f' % acc)

# get the best performing model
model = search.export_model()

# summarize the loaded model
model.summary()

# save the best performing model to file
model.save('model_AI4Sec', overwrite=True, include_optimizer=True)


