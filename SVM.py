from Report import Report
from sklearn.svm import SVC

def SVM(kernel, x_train, y_train, x_test, y_test):    
    svm = SVC(kernel=kernel)
    svm.fit(x_train, y_train)
    predictions = svm.predict(x_test)
    Report(predictions, y_test)
    return svm.score(x_test, y_test)