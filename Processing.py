import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def softmax(z):
    z = np.atleast_2d(z)  # ép về 2D
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def cross_entropy(y, probs, eps=1e-15):
    """
    Multiclass cross-entropy loss
    y: (n,) true labels (0,...,K-1)
    probs: (n,K) predicted probabilities
    """
    probs = np.clip(probs, eps, 1 - eps)
    n = len(y)
    return -np.mean(np.log(probs[np.arange(n), y]))

def predict(X, w):
    z = np.dot(X, w)
    if w.ndim == 1 or w.shape[1] == 1:   # binary
        probs = sigmoid(z).reshape(-1, 1)
        return (probs >= 0.5).astype(int).ravel()
    else:  # multi-class
        probs = softmax(z)
        return np.argmax(probs, axis=1)

def train_test_split_data(write = False):
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
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df['strata']
    )
    if write:
        X_train.to_csv('Data/Train_test/X_train.csv', index=False)
        X_test.to_csv('Data/Train_test/X_test.csv', index=False)
        y_train.to_csv('Data/Train_test/y_train.csv', index=False)
        y_test.to_csv('Data/Train_test/y_test.csv', index=False)

    return X_train, X_test, y_train, y_test


