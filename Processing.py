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


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def train_test_split_data(write=False):
    # Đọc dữ liệu
    df = pd.read_csv('Data/data.csv', sep=';')

    # Chuyển Target sang dạng số
    df['Target_num'] = LabelEncoder().fit_transform(df['Target'])

    # Tách X, y
    X = df.drop(columns=['Target', 'Target_num'])
    y = df['Target_num']

    # Lưu tên cột để dùng sau khi scale
    feature_names = X.columns

    # Tách train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale dữ liệu: fit trên train, transform cả train và test
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Chuyển về DataFrame để lưu CSV
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    y_train_df = pd.DataFrame(y_train)
    y_test_df = pd.DataFrame(y_test)

    # Lưu CSV nếu cần
    if write:
        X_train_df.to_csv('Data/Train_test/X_train.csv', index=False)
        X_test_df.to_csv('Data/Train_test/X_test.csv', index=False)
        y_train_df.to_csv('Data/Train_test/y_train.csv', index=False)
        y_test_df.to_csv('Data/Train_test/y_test.csv', index=False)

    return X_train_df, X_test_df, y_train_df, y_test_df

if __name__ == '__main__':
    train_test_split_data(write=True)




