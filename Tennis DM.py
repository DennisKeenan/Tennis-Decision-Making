import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix,accuracy_score
import graphviz


# Read Data CSV
data=pd.read_csv("PlayTennis.csv")
# print(data.head())
# print(data["outlook"].unique())
# print(data["temp"].unique())
# print(data["humidity"].unique())
# print(data["windy"].unique())
# print(data["play"].unique())

# Histogram
# data["outlook"].hist()
# data["temp"].hist()
# data["humidity"].hist()
# data["windy"].hist()
# data["play"].hist()

# Data Edit (Convert)
le=LabelEncoder()
data["outlook"]=le.fit_transform(data["outlook"])
data["temp"]=le.fit_transform(data["temp"])
data["humidity"]=le.fit_transform(data["humidity"])
data["windy"]=le.fit_transform(data["windy"])
data["play"]=le.fit_transform(data["play"])

# Training
y=data["play"]
x=data.drop(["play"],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
dt=tree.DecisionTreeClassifier()
dt=dt.fit(x_train,y_train)
# tree.plot_tree(dt)
dt2=tree.export_graphviz(dt,out_file=None)
graph=graphviz.Source(dt2)
graph.format="png"
graph.render("tennisdt")

# Model
y_predict=dt.predict(x_test)
print(y_predict==y_test)
