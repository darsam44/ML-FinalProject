import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def KNN(X, y):
    # adaboost
    print("\n**************   KNN   **************")
    sum = 0
    for i in range(1, 100):
        data_train, data_test, cl_train, cl_test = train_test_split(X, y, test_size=.5, random_state=np.random)
        model = KNeighborsClassifier(n_neighbors=4)
        model.fit(data_train, cl_train)
        y_pred5 = model.predict(data_test)
        ac = model.score(data_test, cl_test)
        sum += ac
    print("\nresult: ", sum / 100)
