import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def train_test_split_data():
    df = pd.read_csv('Data/data.csv', sep=';')
    df['Target_num'] = LabelEncoder().fit_transform(df['Target'])

    X = df.drop(columns=['Target', 'Target_num'])
    y = df['Target_num']

    stratifi_list = ["Marital status", "Educational special needs",
                     "Gender", "Unemployment rate", "Nacionality"]

    df['strata'] = df[stratifi_list].astype(str).agg('_'.join, axis=1)

    # Đếm số lượng trong từng nhóm
    counts = df['strata'].value_counts()

    # Giữ lại chỉ những nhóm có >= 2 quan sát
    valid_groups = counts[counts >= 2].index
    df = df[df['strata'].isin(valid_groups)]

    X = df.drop(columns=['Target', 'Target_num'])
    y = df['Target_num']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df['strata']
    )

    X_train.to_csv('Data/Train_test/X_train.csv', index=False)
    X_test.to_csv('Data/Train_test/X_test.csv', index=False)
    y_train.to_csv('Data/Train_test/y_train.csv', index=False)
    y_test.to_csv('Data/Train_test/y_test.csv', index=False)

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = train_test_split_data()
