from Algorithm import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Processing import *
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split_data()

def run_all_cases_test(X_train, y_train, X_test, y_test,
                  lr=0.1, epochs=200, batch_size=32, lam=0.01, momentum=0.9):
    """
    Chạy tất cả các optimizer với 3 penalty: None, L1, L2
    Vẽ biểu đồ loss cho từng combination
    """
    optimizers = ["gd"]
    penalties = [None, "l1", "l2"]
    learning_rates = [0.1, 0.01, 0.001,1,0.5,0.8,1.2, 1.5]
    #learning_rates = [0.5,0.8,1,1.2]
    alpha_0 = [0.1,0.5,0.8,1,1.2,1.5,2,3]
    results = {}

    for opt in optimizers:
        results[opt] = {}
        for pen in penalties:
            print(f"\n--- Running {opt.upper()} with penalty={pen} ---")

            if opt == "gd":
                w,losses = gradient_descent_run_all(X_train, y_train, X_test, y_test,learning_rates, epochs=2000, penalty=pen, lam=0.01)
            elif opt == "backtracking":
                w, losses = backtracking_gd(X_train, y_train, X_test, y_test, alpha_0, epochs=epochs, penalty=pen, lam=lam)
            elif opt == "minibatch":
                w, losses = mini_batch_gd(X_train, y_train, X_test, y_test, learning_rates, epochs=2000, batch_size=32, penalty=pen, lam=0.01)
            elif opt == "sgd":
                w, losses = stochastic_gd(X_train, y_train, X_test, y_test, learning_rates, epochs=100, penalty=pen, lam=0.01)
            elif opt == "newton":
                w, losses = newton_fixed_lr_run_all(X_train, y_train, X_test, y_test, learning_rates, epochs=50, penalty=pen, lam=0.01)
            elif opt == "newton_backtracking":
                w, losses = newton_backtracking_run(X_train, y_train, X_test, y_test, epochs=50, penalty=pen, lam=0.01, alpha=0.25, beta=0.8)
            elif opt == "accelerated":
                w, losses = accelerated_gd_run_all(X_train, y_train, X_test, y_test,learning_rates, epochs=2000, penalty=pen, lam=0.01)
    return results

if __name__ == "__main__":
    results = run_all_cases_test(X_train, y_train, X_test, y_test)