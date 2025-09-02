import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score

# Load data
X_train = pd.read_csv('Data/Train_test/X_train.csv')
X_test = pd.read_csv('Data/Train_test/X_test.csv')
y_train = pd.read_csv('Data/Train_test/y_train.csv').values.ravel()
y_test = pd.read_csv('Data/Train_test/y_test.csv').values.ravel()

def logistics_regression(X_train, X_test, y_train, y_test, n_iter_search=50):
    """
    Huấn luyện Logistic Regression multi-class với RandomizedSearchCV
    cho nhiều penalty và solver. Vẽ hiệu quả theo từng vòng lặp.
    """
    logModel = LogisticRegression(multi_class='multinomial', max_iter=5000)

    param_dist = {
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'C': np.logspace(-4, 4, 50),
        'solver': ['saga'],
        'l1_ratio': [0, 0.25, 0.5, 0.75, 1],  # chỉ dùng khi penalty=elasticnet
        'max_iter': [500, 1000, 2500, 5000]
    }

    clf = RandomizedSearchCV(
        logModel,
        param_distributions=param_dist,
        n_iter=n_iter_search,
        cv=3,
        verbose=1,
        n_jobs=-1,
        scoring='accuracy',
        random_state=42,
        return_train_score=True
    )

    start_time = time.time()
    best_clf = clf.fit(X_train, y_train)
    end_time = time.time()

    best_model = best_clf.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Best Params:", best_clf.best_params_)
    print(f"Accuracy: {acc:.3f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print(f"Time elapsed: {end_time - start_time:.2f} seconds")

    # Plot accuracy vs iteration
    mean_test_scores = best_clf.cv_results_['mean_test_score']
    iterations = np.arange(1, len(mean_test_scores) + 1)

    plt.figure(figsize=(10,6))
    plt.plot(iterations, mean_test_scores, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("CV Accuracy")
    plt.title("RandomizedSearchCV Accuracy per Iteration")
    plt.grid(True)
    plt.savefig("RandomizedSearchCV_accuracy.png")  # save ảnh
    plt.show()

    return best_model, best_clf.best_params_, acc

if __name__ == '__main__':
    logistics_regression(X_train, X_test, y_train, y_test, n_iter_search=50)
