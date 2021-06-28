import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def SVM(X, y):
    #   SVM
    print("\n**************   SVM   **************")
    sum = 0
    for i in range(0, 100):
        data_train, data_test, cl_train, cl_test = train_test_split(X, y, test_size=.5, random_state=np.random)
        ppn = SVC(C=10000, kernel='rbf', degree=3)
        ppn.fit(data_train, cl_train)
        y_pred4 = ppn.predict(data_test)
        acc = ppn.score(data_test, cl_test)
        sum = sum + acc
    print("\nresult: ", sum / 100)
