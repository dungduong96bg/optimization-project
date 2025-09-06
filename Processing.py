import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split


def f1_score(y_true, y_pred, average="macro"):
    """
    Tính F1-score cho binary hoặc multi-class classification.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Nhãn thực tế
    y_pred : array-like, shape (n_samples,)
        Nhãn dự đoán
    average : str, optional (default="macro")
        - "macro": trung bình F1 của từng class (không tính trọng số)
        - "micro": tính dựa trên tổng số TP, FP, FN
        - "weighted": trung bình F1 có trọng số theo số mẫu của mỗi class
        - None: trả về F1 cho từng class

    Returns
    -------
    f1 : float (hoặc dict nếu average=None)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(y_true)

    f1_per_class = {}
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_per_class[c] = f1

    if average is None:
        return f1_per_class
    elif average == "macro":
        return np.mean(list(f1_per_class.values()))
    elif average == "weighted":
        weights = [np.sum(y_true == c) for c in classes]
        return np.average(list(f1_per_class.values()), weights=weights)
    elif average == "micro":
        tp = sum(np.sum((y_true == c) & (y_pred == c)) for c in classes)
        fp = sum(np.sum((y_true != c) & (y_pred == c)) for c in classes)
        fn = sum(np.sum((y_true == c) & (y_pred != c)) for c in classes)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        return 2 * precision * recall / (precision + recall + 1e-8)
    else:
        raise ValueError("average must be one of: None, 'macro', 'micro', 'weighted'")

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

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    train_test_split_data(write=True)




