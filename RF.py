from Report import Report
from sklearn.ensemble import RandomForestClassifier

def RF(criterion, depth, x_train, y_train, x_test, y_test):
    RF = RandomForestClassifier(criterion=criterion, max_depth=depth)
    RF.fit(x_train, y_train)
    predictions = RF.predict(x_test)
    Report(predictions, y_test)
    return RF.score(x_test, y_test)