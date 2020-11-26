from sklearn.datasets import load_iris, load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, explained_variance_score
import matplotlib.pyplot as plt


iris = load_iris()
knn = KNeighborsClassifier()
scores = cross_val_score(knn, iris.data, iris.target, cv=5, scoring='accuracy')
print("Accuracy: {} (+/- {})".format(round(scores.mean(), 2), round(scores.std() * 2, 2)))

parameters = {'n_neighbors': [2, 3, 4, 5]}
clf = GridSearchCV(knn, parameters, cv=5)
clf.fit(iris.data, iris.target)
print(clf.best_params_, clf.best_score_)

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, shuffle=True)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
target_names = ['Setosa', 'Virginica', 'Versicolor']
print(classification_report(y_test, knn.predict(X_test), target_names=target_names))


digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3)
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
conf = confusion_matrix(y_test, predictions)
classes = [str(i) for i in range(10)]
plt.figure()
plt.imshow(conf, cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
plt.xticks(range(len(classes)), classes)
plt.yticks(range(len(classes)), classes)
for i in range(10):
    for j in range(10):
        plt.text(j, i, format(conf[i, j]), horizontalalignment="center")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()
