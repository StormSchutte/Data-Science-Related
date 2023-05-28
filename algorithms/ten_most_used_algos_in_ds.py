"""
I would say that these are probably the 10 most used
algorithms in Data Science
"""


"""
Linear Regression: A statistical algorithm used in machine learning and data
science for predictive analysis.
"""

from sklearn.linear_model import LinearRegression
X, y = [...], [...]
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)

"""
Logistic Regression: This is used when the dependent variable is categorical.
It’s widely used for classification tasks.
"""

from sklearn.linear_model import LogisticRegression
X, y = [...], [...]
model = LogisticRegression()
model.fit(X, y)
predictions = model.predict(X)


"""
K-Means Clustering: An unsupervised learning algorithm for clustering problems.
"""
from sklearn.cluster import KMeans
X = [...]
model = KMeans(n_clusters=3)
model.fit(X)
labels = model.predict(X)


"""
K-Nearest Neighbors (KNN): A simple algorithm used for both classification and
regression problems in supervised learning.
"""
from sklearn.neighbors import KNeighborsClassifier
X, y = [...], [...]
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)
predictions = model.predict(X)


"""
Support Vector Machines (SVM): A classification algorithm used in machine
learning which can also be used for regression, outlier detection, and clustering.
"""
from sklearn import svm
X, y = [...], [...]
model = svm.SVC()
model.fit(X, y)
predictions = model.predict(X)


"""
Decision Trees and Random Forests: Popular in machine learning used for
classification and regression tasks.
"""
from sklearn.ensemble import RandomForestClassifier
X, y = [...], [...]
model = RandomForestClassifier()
model.fit(X, y)
predictions = model.predict(X)


"""
Naive Bayes: Based on the Bayes Theorem, it's used in large scale
classification problems.
"""
from sklearn.naive_bayes import GaussianNB
X, y = [...], [...]
model = GaussianNB()
model.fit(X, y)
predictions = model.predict(X)


"""
Principal Component Analysis (PCA): A dimensionality-reduction algorithm used
in exploratory data analysis and for making predictive models.
"""

from sklearn.decomposition import PCA
X = [...]
model = PCA(n_components=2)
model.fit(X)
X_reduced = model.transform(X)



"""
Gradient Descent: An optimization algorithm used to minimize some function by
iteratively moving in the direction of steepest descent.
"""
from sklearn.linear_model import SGDRegressor
X, y = [...], [...]
model = SGDRegressor()
model.fit(X, y)
predictions = model.predict(X)


"""
Deep Learning Algorithms: Algorithms related to artificial neural networks,
convolutional neural networks (CNN), recurrent neural networks (RNN), and
others for tasks in image and speech recognition, natural language processing
and more.
"""
from keras.models import Sequential
from keras.layers import Dense
X, y = [...], [...]
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=150, batch_size=10)
predictions = model.predict(X)
