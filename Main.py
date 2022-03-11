import argparse
import numpy as np
from PreProcess import preProcessAndSplit
from SVM import SVM
from RF import RF
from DT import DT

def main(args):
    x_train, y_train, x_test, y_test = preProcessAndSplit(args.path)
    if args.model == 'svm':
        SVM(args.kernel, x_train, y_train, x_test, y_test)
    elif args.model == 'rf':
        RF(args.criterion, args.max_depth, x_train, y_train, x_test, y_test)
    elif args.model == 'tree':
        DT(args.criterion, args.max_depth, x_train, y_train, x_test, y_test)
    elif args.model == 'all':
        accu = []
        accu.append(SVM(args.kernel, x_train, y_train, x_test, y_test))
        accu.append(RF(args.criterion, args.max_depth, x_train, y_train, x_test, y_test))
        accu.append(DT(args.criterion, args.max_depth, x_train, y_train, x_test, y_test))
        if np.argmax(accu) == 0:
            print(f"O melhor modelo foi o svm com a accuracia de {accu[0]}")
        elif np.argmax(accu) == 1:
            print(f"O melhor modelo foi o Random Forest com a accuracia de {accu[1]}")
        elif np.argmax(accu) == 2:
            print(f"O melhor modelo foi a Árvore de Decisão com a accuracia de {accu[2]}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./TravelInsurancePrediction.csv')
    parser.add_argument('--model', type=str, default='all')
    parser.add_argument('--kernel', type=str, default='rbf')
    parser.add_argument('--criterion', type=str, default='gini')
    parser.add_argument('--max_depth', type=int, default=None)


    args = parser.parse_args()
    main(args)