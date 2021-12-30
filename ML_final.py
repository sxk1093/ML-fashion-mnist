import numpy as np
import matplotlib.pyplot as plt
import mnist_reader

from sklearn import neighbors, datasets, metrics, svm
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import plot_confusion_matrix

#loading teh fashion mnist data
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
print("Data is read. \n")

#Standardising the dataset
scaler = StandardScaler()
#Fit on training set only.
scaler.fit(X_train)
#Apply transform to both the training set and the test set.
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#Make an instance of the Model, 95% of the variance
pca = PCA(.95)
pca.fit(X_train)
#Apply the mapping (transform) to both the training set and the test set.
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
print("PCA is implemented. \n")

#Defining the algorithms
def K_nn(X_train, X_test, y_train, y_test, n):
    knn = KNeighborsClassifier(n_neighbors=n)

    knn.fit(X_train, y_train)
    print("Knn model trained. \n")

    #confusion matrix
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    titles_options = [("K-nn confusion matrix", 'true')]

    for title, normalize in titles_options:
        disp = plot_confusion_matrix(knn, X_test, y_test,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)
    plt.show()

    #back to normal stuff

    y_pred = knn.predict(X_test)
    return y_pred
    
def Rand_Forest(X_train, X_test, y_train, y_test, n):
    clf = RandomForestClassifier(n_estimators=n, random_state=0)

    clf.fit(X_train, y_train)
    print("Random forest model trained. \n")

    #confusion matrix
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    titles_options = [("Random forest confusion matrix", 'true')]

    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, X_test, y_test,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)
    plt.show()

    #back to normal stuff

    y_pred = clf.predict(X_test)
    return y_pred

def SVM(X_train, X_test, y_train, y_test, k, c):

    clf = svm.SVC(kernel=k, C=c)

    clf.fit(X_train, y_train)
    print("Support vector machine trained. \n")

    #confusion matrix
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    titles_options = [("SVM confusion matrix", 'true')]

    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, X_test, y_test,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)
    plt.show()

    #back to normal stuff
    y_pred = clf.predict(X_test)
    return y_pred


print("tests on K-nn")
y_pred = K_nn(X_train, X_test, y_train, y_test, 6)
print("Accuracy n=6:",metrics.accuracy_score(y_test, y_pred))

print("tests on Random forrests")
y_pred = Rand_Forest(X_train, X_test, y_train, y_test, 500)
print("Accuracy n=500:",metrics.accuracy_score(y_test, y_pred))

print("tests on svm")
y_pred = SVM(X_train, X_test, y_train, y_test, "rbf", 13)
print("Accuracy c=13:",metrics.accuracy_score(y_test, y_pred))
