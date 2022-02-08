import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

file_path = r"C:\Users\Simon\PycharmData\heart\heart.csv"
df = pd.read_csv(file_path)
df.hist(figsize=(30,30))
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")

# create a dataframe with training data expect Outcome column
X = df.drop(columns=['target'])

y = df['target'].values

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors=17)

# Fit the classifier to the data
knn.fit(X_train, y_train)

# show first 5 model predictions on the test data
print(knn.predict(X_test)[0:20])
print(y_test[0:20])

print(knn.score(X_test, y_test))
