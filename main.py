import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# data modeling
import os
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, classification_report
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import seaborn as sns

from sklearn import datasets
from sklearn import tree

# class KNN:
#     def __init__(self, k=3):
#         self.k = k
#
#     def fit(self, X, y):
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234, stratify=y)
#         self.X_train = X_train
#         self.X_test = X_test
#         self.y_train = y_train
#         self.y_test = y_test
#         pass
#
#     def predict(self, X):
#         pass

# import dataset
file_path = r"C:\Users\Simon\PycharmData\heart\heart.csv"
df = pd.read_csv(file_path)
df.hist(figsize=(30,30))
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")

# create a dataframe with training data expect Outcome column
#X = df.drop(columns=['target'])
#y = df['target'].values
y = df["target"]
X = df.drop('target',axis=1)
# Z = df.drop('target')

# split dataset into test and train data. The size of the test data can be altered
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=4)


# print(X_train.shape)

# Create KNN classifier
# knn = KNeighborsClassifier(n_neighbors=10)

# Fit the classifier to the data
# knn.fit(X_train, y_train)

# show first 5 model predictions on the test data
# print(knn.predict(X_test)[0:20])
# print(y_test[0:20])

# print(knn.score(X_test, y_test))

m5 = 'K-NeighborsClassifier'
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
knn_predicted = knn.predict(X_test)
knn_conf_matrix = confusion_matrix(y_test, knn_predicted)
knn_acc_score = accuracy_score(y_test, knn_predicted)
print("confusion matrix")
print(knn_conf_matrix)
print("\n")
print("Accuracy of K-NeighborsClassifier:", knn_acc_score * 100, '\n')
print(classification_report(y_test, knn_predicted))


error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed',
         marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate)))

acc = []
# Will take some time

for i in range(1, 40):
    neigh = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    acc.append(metrics.accuracy_score(y_test, yhat))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), acc, color='blue', linestyle='dashed',
         marker='o', markerfacecolor='red', markersize=10)
plt.title('Accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
print("Maximum accuracy:-", max(acc), "at K =", acc.index(max(acc)))

m6 = 'DecisionTreeClassifier'
dt = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=6)
dt.fit(X_train, y_train)
dt_predicted = dt.predict(X_test)
dt_conf_matrix = confusion_matrix(y_test, dt_predicted)
dt_acc_score = accuracy_score(y_test, dt_predicted)

text_representation = tree.export_text(dt)
print(text_representation)
with open("decistion_tree.log", "w") as fout:
    fout.write(text_representation)

fig = plt.figure(figsize=(50,7), dpi=350)
_ = tree.plot_tree(dt,
                    fontsize=7,
                   feature_names=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'],
                   class_names=['has heart disease','no has heart disease'],
                   filled=True, rounded=True)

# from dtreeviz.trees import dtreeviz # remember to load the package
#
# viz = dtreeviz(dt,
#              target_name="target",
#              feature_names=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'],
#              class_names=['has', 'not has'])
# viz.view()
# viz.save("decision_tree.png")

print("confusion matrix")
print("\n")
print("Accuracy of DecisionTreeClassifier:", dt_acc_score * 100, '\n')
print(classification_report(y_test, dt_predicted))

error_rate = []
for i in range(1,40):
    dt = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=i)
    dt.fit(X_train, y_train)
    pred_i = dt.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed',
         marker='o',markerfacecolor='red', markersize=3)
plt.title('Error Rate vs. max_depth Value (DecisionTreeClassifier)')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()
