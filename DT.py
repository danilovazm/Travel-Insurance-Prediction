from Report import Report
from sklearn.tree import DecisionTreeClassifier

def DT(criterion, max_depth, x_train, y_train, x_test, y_test):
    DT = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    DT.fit(x_train, y_train)
    predictions = DT.predict(x_test)
    Report(predictions, y_test)
    return DT.score(x_test, y_test)