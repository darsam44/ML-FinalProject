import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def logistic(X, y):
    print("\n**************   LogisticRegression   **************")
    sum = 0
    for i in range(1, 100):
        data_train, data_test, cl_train, cl_test = train_test_split(X, y, test_size=.5, random_state=np.random)
        # logistic regression
        model = LogisticRegression()
        model.fit(data_train, cl_train)
        y_pred1 = model.predict(data_test)
        ac = model.score(data_test, cl_test)
        sum += ac
    print("\nresult: ", sum / 100)
