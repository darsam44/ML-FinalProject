import numpy as np
from sklearn.model_selection import train_test_split


def adaboost(X, y):
    # adaboost
    print("\n**************   ADABOOST   **************")
    sum = 0
    for i in range(0, 100):
        from sklearn.ensemble import AdaBoostClassifier
        data_train, data_test, cl_train, cl_test = train_test_split(X, y, test_size=.5, random_state=np.random)
        model = AdaBoostClassifier()
        model.fit(data_train, cl_train)
        y_pred3 = model.predict(data_test)
        ac = model.score(data_test, cl_test)
        sum += ac
        # print(ac)
    print("\nresult: ", sum / 100)
