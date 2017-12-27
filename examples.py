import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot
import xgboost as xb
from xgboost import plot_tree
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel


#Example 1 - Numerical input data
def diabetes_ex(save = True, load = False):

	if load:
		model = pickle.load(open("pima.pickle.dat", "rb"))
	else:
		dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
		
		X = dataset[:,0:8]
		Y = dataset[:,8]
		
		seed = 7
		test_size = 0.2
		X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size,
		random_state=seed)
		
		model = xb.XGBClassifier()
		model.fit(X_train, y_train)
	
		if save:
			pickle.dump(model, open("pima.pickle.dat", "wb"))
			print("Saved model to: pima.pickle.dat")
		
	y_pred = model.predict(X_test)
	predictions = [round(value) for value in y_pred]

	accuracy = accuracy_score(y_test, predictions)
	return (print("Accuracy: %.2f%%" % (accuracy * 100.0)))
	
	
#Example 2 - Categorical labels
def iris_ex(importance, show_tree = True):

	'''This example shows how to handle non-numerical classification
	labels, by using the LabelEncoder from Numpy. Furthermore it uses a
	helper function to plot a decision tree from the algorithm
	'''

	dataset = pd.read_csv('iris.csv', header = None)
	dataset = dataset.values
	
	X = dataset[:,0:4]
	Y = dataset[:,4]
	
	label_encoder = LabelEncoder().fit(Y)
	Y = label_encoder.transform(Y)
	
	seed = 7
	test_size = 0.33
	X_train, X_test, y_train, y_test = train_test_split(X, Y,
	test_size=test_size, random_state=seed)

	model = xb.XGBClassifier(max_depth = 2)
	model.fit(X_train, y_train)
	print(model)
	
	y_pred = model.predict(X_test)
	predictions = [round(value) for value in y_pred]

	accuracy = accuracy_score(y_test, predictions)
	print("Accuracy: %.2f%%" % (accuracy * 100.0))
	
	if show_tree is True:
		plot_tree(model, num_trees=1)
		pyplot.show()
	
	#Example of visualizing importance of features based on decision trees in the algo
	if importance is 'basic':	
		pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
		pyplot.show()
	elif importance is 'advanced':
		plot_importance(model)
		pyplot.show()
	else:
		print("You must specify 'basic' or 'advanced' visualization of features")
	
	#Sorting features in terms of importance - returns a list
	thresholds = np.sort(model.feature_importances_)
	
	for thresh in thresholds:
		# select features using threshold
		selection = SelectFromModel(model, threshold=thresh, prefit=True)
		select_X_train = selection.transform(X_train)
		# train model
		selection_model = xb.XGBClassifier()
		selection_model.fit(select_X_train, y_train)
		# eval model
		select_X_test = selection.transform(X_test)
		y_pred = selection_model.predict(select_X_test)
		predictions = [round(value) for value in y_pred]
		accuracy = accuracy_score(y_test, predictions)
		print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1],
		accuracy*100.0))

iris_ex('advanced')


#Example 3 - Purely categorical input
def breast_ex():

	'''This examples shows how you can handle purely non-numerical data
	from your dataset, and turn it into numerical data that Numpy accepts.
	This variation is a one-hot encoding, but you may use any approach you want
	'''

	data = read_csv('datasets-uci-breast-cancer.csv', header=None)
	dataset = data.values

	X = dataset[:,0:9]
	X = X.astype(str)
	Y = dataset[:,9]

	columns = []
	for i in range(0, X.shape[1]):
		label_encoder = LabelEncoder()
		feature = label_encoder.fit_transform(X[:,i])
		feature = feature.reshape(X.shape[0], 1)
		onehot_encoder = OneHotEncoder(sparse=False)
		feature = onehot_encoder.fit_transform(feature)
		columns.append(feature)

	encoded_x = column_stack(columns)
	print("X shape: : ", encoded_x.shape)

	label_encoder = LabelEncoder()
	label_encoder = label_encoder.fit(Y)
	label_encoded_y = label_encoder.transform(Y)

	seed = 7
	test_size = 0.33
	X_train, X_test, y_train, y_test = train_test_split(encoded_x, label_encoded_y,
	test_size=test_size, random_state=seed)

	model = xb.XGBClassifier()
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)
	predictions = [round(value) for value in y_pred]

	accuracy = accuracy_score(y_test, predictions)
	print("Accuracy: %.2f%%" % (accuracy * 100.0))
	
	
#Example 4 - Missing data handling
def horse_ex():

	'''This example shows different ways to handle missing data in a 
	dataset. Normally you would of course need to explore the dataset
	beforehand to know how missing values are represented. Furthermore
	it shows how to use k-fold and cross validation as evaluation metric'''
	
	dataset = pd.read_csv('horse-colic.csv', delim_whitespace = True,
							header =  None)
	dataset = dataset.values
	
	X = dataset[:,0:27]
	Y = dataset[:,27]
	
	#Manual error handling 
	#X[X == '?'] = 9**10
	X[X == '?'] = np.nan
	X.astype(np.float32)
	
	#Automatic error handling
	#imputer = Imputer()
	#X = imputer.fit_transform(X)
	
	label_encoder = LabelEncoder().fit(Y)
	Y = label_encoder.transform(Y)
	Y.astype(int)
	
	model = xb.XGBClassifier(max_depth=2) #max depth has a large impact on accuracy for this example +-6%
		
	#Using k-fold and cross validation for evaluating 
	kfold = KFold(n_splits=10, random_state=7)
	results = cross_val_score(model, X, Y, cv=kfold)
	print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
	
horse_ex()
