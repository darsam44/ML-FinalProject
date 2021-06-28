import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def DecisionTree(X, y):
    sum = 0
    print("\n**************   DecisionTree   **************")
    for i in range(0, 100):
        data_train, data_test, cl_train, cl_test = train_test_split(X, y, test_size=.5, random_state=np.random)
        # decision tree
        model = DecisionTreeClassifier(random_state=1)
        model.fit(data_train, cl_train)
        y_pred1 = model.predict(data_test)
        ac = model.score(data_test, cl_test)
        sum += ac
    print("\nresult: ", sum / 100)
