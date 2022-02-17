import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# data modeling
# import os
from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix, roc_curve, classification_report
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn import metrics
# import seaborn as sns

# from sklearn import datasets
# from sklearn import tree

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

# df.hist(figsize=(30,30))
# corrmat = df.corr()
# top_corr_features = corrmat.index
#
# plt.figure(figsize=(20,20))
# g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")

# create a dataframe with training data expect Outcome column
# X = df.drop(columns=['target'])
# y = df['target'].values
y = df["target"]
X = df.drop('target', axis=1)
# Z = df.drop('target')

# split dataset into test and train data. The size of the test data can be altered
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=4)

print(df.head, '\n')

# print(X_train.shape)

# Create KNN classifier
# knn = KNeighborsClassifier(n_neighbors=10)

# Fit the classifier to the data
# knn.fit(X_train, y_train)

# show first 5 model predictions on the test data
# print(knn.predict(X_test)[0:20])
# print(y_test[0:20])

# print(knn.score(X_test, y_test))

# m5 = 'K-NeighborsClassifier'
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
knn_predicted = knn.predict(X_test)
knn_acc_score = accuracy_score(y_test, knn_predicted)
print("KNN: ", '\n')
print("N_Neighbors: ", knn.n_neighbors)
print("Accuracy:", knn_acc_score * 100, '\n')
# ML print(classification_report(y_test, knn_predicted))


# ML error_rate = []
# ML for i in range(1,40):
# ML     knn = KNeighborsClassifier(n_neighbors=i)
# ML     knn.fit(X_train,y_train)
# ML     pred_i = knn.predict(X_test)
# ML     error_rate.append(np.mean(pred_i != y_test))
# ML
# ML plt.figure(figsize=(10,6))
# ML plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed',
# ML          marker='o',markerfacecolor='red', markersize=10)
# ML plt.title('Error Rate vs. K Value')
# ML plt.xlabel('K')
# ML plt.ylabel('Error Rate')
# ML print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate)))
# ML
# ML acc = []
# ML # Will take some time
# ML
# ML for i in range(1, 40):
# ML     neigh = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
# ML     yhat = neigh.predict(X_test)
# ML     acc.append(metrics.accuracy_score(y_test, yhat))
# ML
# ML plt.figure(figsize=(10, 6))
# ML plt.plot(range(1, 40), acc, color='blue', linestyle='dashed',
# ML          marker='o', markerfacecolor='red', markersize=10)
# ML plt.title('Accuracy vs. K Value')
# ML plt.xlabel('K')
# ML plt.ylabel('Accuracy')
# ML print("Maximum accuracy:-", max(acc), "at K =", acc.index(max(acc)))


dt = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=6)
dt.fit(X_train, y_train)
dt_predicted = dt.predict(X_test)
dt_acc_score = accuracy_score(y_test, dt_predicted)

print("Decision Tree:", '\n')
print("Depth:", dt.max_depth)
print("Accuracy:", dt_acc_score * 100, '\n')
# print(classification_report(y_test, dt_predicted))

# ML text_representation = tree.export_text(dt)
# ML print(text_representation)
# ML with open("decistion_tree.log", "w") as fout:
# ML     fout.write(text_representation)

# ML fig = plt.figure(figsize=(50,7), dpi=350)
# ML _ = tree.plot_tree(dt,
# ML                     fontsize=7,
# ML                    feature_names=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'],
# ML                    class_names=['has heart disease','no has heart disease'],
# ML                    filled=True, rounded=True)

# from dtreeviz.trees import dtreeviz # remember to load the package
#
# viz = dtreeviz(dt,
#              target_name="target",
#              feature_names=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'],
#              class_names=['has', 'not has'])
# viz.view()
# viz.save("decision_tree.png")

# ML error_rate = []
# ML for i in range(1,40):
# ML     dt = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=i)
# ML     dt.fit(X_train, y_train)
# ML     pred_i = dt.predict(X_test)
# ML     error_rate.append(np.mean(pred_i != y_test))
# ML plt.figure(figsize=(10,6))
# ML plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed',
# ML          marker='o',markerfacecolor='red', markersize=3)
# ML plt.title('Error Rate vs. max_depth Value (DecisionTreeClassifier)')
# ML plt.xlabel('K')
# ML plt.ylabel('Error Rate')
# ML plt.show()
