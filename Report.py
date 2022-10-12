import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

def Report(name, predictions, labels):
    cf_matrix = confusion_matrix(labels, predictions)
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
    ax.set_title(f'Confusion Matrix {name}\n')
    ax.set_xlabel('Predictions')
    ax.set_ylabel('Labels')
    plt.savefig(name + ".png")
    plt.show()
    print(name)
    print(classification_report(labels, predictions) + '\n')
