import pandas as pd
from sklearn.model_selection import train_test_split

def catToNumber(df):
    categorical_cols=df.select_dtypes(include=object).columns.to_list()
    for col in categorical_cols:
        df[col]=df[col].astype('category').cat.codes
    return df

def preProcessAndSplit(path):
    df = pd.read_csv(path)
    df = df.sample(frac=1)
    df = catToNumber(df)
    dfX = df.iloc[:,:-1]
    dfY = df['TravelInsurance']
    x_train, x_test, y_train, y_test = train_test_split(dfX, dfY, test_size=0.2,stratify=dfY)
    return x_train, y_train, x_test, y_test