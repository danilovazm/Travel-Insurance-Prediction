import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

def Report(predictions, labels):
    cf_matrix = confusion_matrix(labels, predictions)
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
    ax.set_title('Confusion Matrix\n')
    ax.set_xlabel('Predictions')
    ax.set_ylabel('Labels')
    plt.show()
    print(classification_report(labels, predictions))
