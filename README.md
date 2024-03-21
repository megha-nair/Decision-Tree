# Decision-Tree
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.tree import  DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data=pd.read_csv('drug200.csv')
df=data.copy()
data
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Sex"]=le.fit_transform(data["Sex"])
data["BP"]=le.fit_transform(data["BP"])
data["Cholesterol"]=le.fit_transform(data["Cholesterol"])
data["Drug"]=le.fit_transform(data["Drug"])
data
x=data.drop(["Drug"],axis=1)
x
y=data.Drug
y
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.2,random_state=45)
tre=DecisionTreeClassifier()
tre.fit(xtrain,ytrain)
ypred=tre.predict(xtest)
ypred
from sklearn.metrics import accuracy_score
accuracy_score(ytest,ypred)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest,ypred)
cm
sns.heatmap(cm,annot=True)
from matplotlib import pyplot as plt
from sklearn import tree
_=tree.plot_tree(tre,
                  feature_names=x.columns,
                  class_names=df.Drug.unique(),
                  filled = True
                  )


