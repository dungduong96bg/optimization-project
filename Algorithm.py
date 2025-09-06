import numpy as np
import matplotlib.pyplot as plt
from Processing import predict,softmax,cross_entropy,f1_score
from sklearn.metrics import accuracy_score
import time
import pandas as pd
# Hàm sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
# Hàm loss với penalty
def logistic_loss(X, y, w, penalty=None, lam=0.01):
    """
    Logistic loss (cross-entropy) với auto-balance class weight.
    Tự động gán trọng số theo công thức w_c = N / (K * n_c).

    Parameters
    ----------
    X : ndarray, shape (m, d)
        Ma trận dữ liệu.
    y : ndarray, shape (m,)
        Nhãn (0..K-1).
    w : ndarray, shape (d, K)
        Trọng số mô hình.
    penalty : str, optional ("l1", "l2")
        Loại regularization.
    lam : float
        Hệ số regularization.

    Returns
    -------
    loss : float
    """
    m = len(y)
    logits = X @ w
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # tránh overflow
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # One-hot encode y
    K = probs.shape[1]
    y_onehot = np.eye(K)[y]

    # --- AUTO-BALANCE ---
    class_counts = np.bincount(y, minlength=K)
    weights = m / (K * class_counts)   # w_c = N / (K * n_c)
    sample_weights = weights[y]

    # Weighted cross-entropy
    loss = -np.sum(sample_weights * np.sum(y_onehot * np.log(probs + 1e-8), axis=1)) / np.sum(sample_weights)

    # Regularization
    if penalty == "l2":
        loss += lam * np.sum(w ** 2) / (2 * m)
    elif penalty == "l1":
        loss += lam * np.sum(np.abs(w)) / (2 * m)

    return loss

# Gradient của loss
def gradient(X, y, w, penalty=None, lam=0.01):
    if isinstance(X, pd.DataFrame):
        X = X.values

    m = len(y)
    logits = X @ w                # shape: (m, 3)
    probs = softmax(logits)       # shape: (m, 3)

    y = np.array(y).astype(int)
    y_onehot = np.eye(probs.shape[1])[y]  # shape: (m, 3)

    grad = (X.T @ (probs - y_onehot)) / m  # shape: (n_features, 3)

    if penalty == "l2":
        grad += lam * w / m
    elif penalty == "l1":
        grad += lam * np.sign(w) / m

    return grad

def gradient_descent(X, y, lr=0.01, epochs=1000, penalty=None, lam=0.01, tol=1e-3):
    """
    Gradient Descent cơ bản, lưu lại toàn bộ history.
    """
    w = np.zeros((X.shape[1], len(np.unique(y))))
    history_w = []
    losses_epoch = []
    times = []

    start_time = time.time()

    for epoch in range(epochs):
        grad = gradient(X, y, w, penalty, lam)
        w -= lr * grad

        # Lưu loss và weight
        losses_epoch.append(logistic_loss(X, y, w, penalty, lam))
        history_w.append(w.copy())
        times.append(time.time() - start_time)

        if np.linalg.norm(grad) < tol:
            break

    return w, history_w, losses_epoch, times

def backtracking_gd(X_train, y_train, X_test, y_test, alpha_values, epochs=1000, penalty=None, lam=0.01, beta=0.8, tol=0.005):
    """
    Chạy Backtracking GD với nhiều alpha khác nhau.
    Vẽ loss curve, so sánh accuracy, chọn alpha tốt nhất.
    """
    results = {}
    best_acc = -1
    best_alpha = None
    best_model = None

    plt.figure(figsize=(8,6))

    for alpha in alpha_values:
        w = np.zeros((X_train.shape[1], len(np.unique(y_train))))
        losses = []
        for _ in range(epochs):
            grad = gradient(X_train, y_train, w, penalty, lam)
            if np.linalg.norm(grad) < tol:   # dừng sớm nếu gradient nhỏ
                break
            t = 1
            while logistic_loss(X_train, y_train, w - t*grad, penalty, lam) > \
                  logistic_loss(X_train, y_train, w, penalty, lam) - alpha*t*np.linalg.norm(grad)**2:
                t *= beta
            w -= t * grad
            losses.append(logistic_loss(X_train, y_train, w, penalty, lam))

        # Evaluate
        y_pred = predict(X_test, w)
        acc = accuracy_score(y_test, y_pred)
        results[alpha] = {"weights": w, "losses": losses, "accuracy": acc}
        plt.plot(losses, label=f"alpha={alpha} (acc={acc:.3f})")

        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha
            best_model = {"weights": w, "losses": losses, "accuracy": acc}

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Backtracking GD: Loss curves for different alphas (penalty={penalty})")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"\nBest alpha: {best_alpha}, Accuracy: {best_acc:.4f}")

    return results, best_model

def mini_batch_gd(X_train, y_train, X_test, y_test, lr_values, epochs=1000, batch_size=32, penalty=None, lam=0.01):
    """
    Chạy Mini-batch GD với nhiều learning rate.
    Vẽ loss curve cho từng lr, so sánh accuracy, chọn lr tốt nhất.
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    results = {}
    best_acc = -1
    best_lr = None
    best_model = None

    plt.figure(figsize=(10,6))

    for lr in lr_values:
        w = np.zeros((X_train.shape[1], len(np.unique(y_train))))
        losses = []
        start_time = time.time()
        times = []

        for epoch in range(epochs):
            # Shuffle dữ liệu
            idx = np.random.permutation(len(y_train))
            X_shuffled, y_shuffled = X_train[idx], y_train[idx]

            # Mini-batch
            for i in range(0, len(y_train), batch_size):
                X_batch, y_batch = X_shuffled[i:i+batch_size], y_shuffled[i:i+batch_size]
                grad = gradient(X_batch, y_batch, w, penalty, lam)
                w -= lr * grad

            # Tính loss và thời gian
            loss = logistic_loss(X_train, y_train, w, penalty, lam)
            losses.append(loss)
            times.append(time.time() - start_time)

        # Evaluate trên test set
        y_pred= predict(X_test, w)
        acc = accuracy_score(y_test, y_pred)

        results[lr] = {"weights": w, "losses": losses, "accuracy": acc, "times": times}

        plt.plot(losses, label=f"lr={lr} (acc={acc:.3f})")

        if acc > best_acc:
            best_acc = acc
            best_lr = lr
            best_model = {"weights": w, "losses": losses, "accuracy": acc, "times": times}

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Mini-batch GD: Loss curves for different learning rates (penalty={penalty})")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"\nBest learning rate: {best_lr}, Accuracy: {best_acc:.4f}")

    return results, best_model

def stochastic_gd( X_train, y_train, X_test, y_test, lr_values,epochs=100, penalty=None, lam=0.01, print_every=10):
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    results = {}
    best_acc = -1
    best_lr = None
    best_model = None

    plt.figure(figsize=(10, 6))

    for lr in lr_values:
        w = np.zeros((X_train.shape[1], len(np.unique(y_train))))
        losses = []
        start_time = time.time()
        times = []

        for epoch in range(epochs):
            # Shuffle index
            idx = np.random.permutation(len(y_train))

            # SGD: cập nhật từng mẫu
            for i in idx:
                xi = X_train[i:i + 1]
                yi = y_train[i:i + 1]
                grad = gradient(xi, yi, w, penalty, lam)
                w -= lr * grad

            # Tính loss cuối epoch
            loss = logistic_loss(X_train, y_train, w, penalty, lam)
            losses.append(loss)
            times.append(time.time() - start_time)

            if epoch % print_every == 0:
                print(f"[lr={lr}] Epoch {epoch}, Loss={loss:.4f}")

        # Evaluate test
        y_pred = predict(X_test, w)
        acc = accuracy_score(y_test, y_pred)

        results[lr] = {
            "weights": w, "losses": losses,
            "accuracy": acc, "times": times
        }

        plt.plot(losses, label=f"lr={lr} (acc={acc:.3f})")

        if acc > best_acc:
            best_acc = acc
            best_lr = lr
            best_model = results[lr]

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Stochastic GD (batch_size=1), penalty={penalty}")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"\nBest learning rate: {best_lr}, Accuracy: {best_acc:.4f}")

    return results, best_model

def newton_method(X, y, lr=0.1, epochs=50, penalty=None, lam=0.01):
    """
    Newton's method with learning rate (binary logistic regression version)
    """
    w = np.zeros(X.shape[1])
    losses = []

    for _ in range(epochs):
        logits = X @ w
        probs = 1 / (1 + np.exp(-logits))  # sigmoid

        grad = (X.T @ (probs - y)) / len(y)

        # Hessian
        W = np.diag(probs * (1 - probs))
        H = (X.T @ W @ X) / len(y)

        step = np.linalg.pinv(H) @ grad
        w -= lr * step

        losses.append(logistic_loss(X, y, w, penalty, lam))

    return w, losses

def newton_backtracking(X, y, epochs=50, penalty=None, lam=0.01, alpha=0.25, beta=0.8):
    """
    Newton method với backtracking line search.
    - alpha: hệ số Armijo (0 < alpha < 0.5)
    - beta: hệ số giảm step size (0 < beta < 1)
    """
    w = np.zeros((X.shape[1], len(np.unique(y))))
    losses = []

    for _ in range(epochs):
        logits = X @ w
        probs = softmax(logits)
        y_onehot = np.eye(probs.shape[1])[y]

        # Gradient
        grad = (X.T @ (probs - y_onehot)) / len(y)

        # Thêm regularization
        if penalty == 'l2':
            grad += lam * w
        elif penalty == 'l1':
            grad += lam * np.sign(w)

        # Hessian (xấp xỉ block-diagonal)
        W = np.zeros((len(y), len(y)))
        for i in range(len(y)):
            p = probs[i]
            W[i, i] = np.sum(p * (1 - p))  # gần đúng
        H = (X.T @ W @ X) / len(y)

        # Newton direction
        try:
            d = -np.linalg.solve(H + 1e-6*np.eye(H.shape[0]), grad)  # regularized solve
        except np.linalg.LinAlgError:
            d = -grad  # fallback: gradient descent step

        # Backtracking line search
        t = 1.0
        loss_curr = cross_entropy(y, probs)
        while True:
            w_new = w + t * d
            logits_new = X @ w_new
            probs_new = softmax(logits_new)
            loss_new = cross_entropy(y, probs_new)

            if loss_new <= loss_curr + alpha * t * np.sum(grad * d):
                break
            t *= beta
            if t < 1e-8:
                break

        # Update
        w = w + t * d
        losses.append(loss_new)

    return w, losses

def accelerated_gd(X, y, lr=0.01, epochs=1000, penalty=None, lam=0.01,
                   momentum=0.9, tol=1e-3):
    w = np.zeros((X.shape[1], len(np.unique(y))))
    v = np.zeros_like(w)

    history_w = []
    train_losses = []
    times = []

    start = time.time()
    for _ in range(epochs):
        grad = gradient(X, y, w - momentum*v, penalty, lam)
        v = momentum * v + lr * grad
        w -= v

        # lưu loss, w và thời gian
        train_losses.append(logistic_loss(X, y, w, penalty, lam))
        history_w.append(w.copy())
        times.append(time.time() - start)

        if np.linalg.norm(grad) < tol:
            break

    return w, history_w, train_losses, times

def gradient_descent_run_all(X_train, y_train, X_val, y_val,
                             learning_rates,
                             epochs=1000,
                             penalty=None,
                             lam=0.01,
                             tol=1e-3):
    results = {}
    best_f1 = -1
    best_lr = None
    best_model = None

    for lr in learning_rates:
        print(f"\n🔹 Training with lr={lr}, penalty={penalty}...")

        # Train bằng gradient descent (phiên bản có lưu history_w)
        w, history_w, train_losses, times = gradient_descent(
            X_train, y_train,
            lr=lr, epochs=epochs,
            penalty=penalty, lam=lam, tol=tol
        )

        # --- Evaluate trên Validation ---
        val_losses_epoch, val_acc_epoch, val_f1_epoch = [], [], []
        for w_epoch in history_w:
            val_losses_epoch.append(logistic_loss(X_val, y_val, w_epoch, penalty, lam))
            y_val_pred = predict(X_val, w_epoch)
            val_acc_epoch.append(accuracy_score(y_val, y_val_pred))
            val_f1_epoch.append(f1_score(y_val, y_val_pred, average="macro"))

        # Epoch tối ưu (theo val_loss)
        epoch_opt = int(np.argmin(val_losses_epoch))
        w_opt = history_w[epoch_opt]

        # Metrics tại epoch tối ưu
        y_val_pred_opt = predict(X_val, w_opt)
        acc_opt = accuracy_score(y_val, y_val_pred_opt)
        f1_opt = f1_score(y_val, y_val_pred_opt, average="macro")

        # Metrics tại epoch cuối
        y_val_pred_final = predict(X_val, w)
        acc_final = accuracy_score(y_val, y_val_pred_final)
        f1_final = f1_score(y_val, y_val_pred_final, average="macro")

        results[lr] = {
            "weights": w,
            "history_w": history_w,
            "train_losses": train_losses,
            "val_losses_epoch": val_losses_epoch,
            "val_acc_epoch": val_acc_epoch,
            "val_f1_epoch": val_f1_epoch,
            "times": times,
            "epoch_optimal": epoch_opt + 1,
            "acc_at_opt_epoch": acc_opt,
            "f1_at_opt_epoch": f1_opt,
            "acc_final": acc_final,
            "f1_final": f1_final,
            "time_opt_epoch": times[epoch_opt],
            "time_total_epoch": times[-1]
        }

        # Best model theo F1 tại epoch tối ưu
        if f1_opt > best_f1:
            best_f1 = f1_opt
            best_lr = lr
            best_model = results[lr]

    # --- Learning Curve: Train vs Val Loss ---
    plt.figure(figsize=(10,6))
    for lr, info in results.items():
        epochs_range = range(1, len(info["train_losses"]) + 1)
        plt.plot(epochs_range, info["train_losses"], label=f"Train lr={lr}")
        plt.plot(epochs_range, info["val_losses_epoch"], "--", label=f"Val lr={lr}")

        # Đánh dấu epoch tối ưu
        epoch_opt = info["epoch_optimal"]
        val_loss_opt = info["val_losses_epoch"][epoch_opt-1]
        plt.scatter(epoch_opt, val_loss_opt, color="red", zorder=5)
        plt.text(epoch_opt, val_loss_opt, f"opt={epoch_opt}", fontsize=8, color="red")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Learning Curve: Train vs Val Loss (penalty={penalty})")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Learning Curve: Validation F1 ---
    plt.figure(figsize=(10,6))
    for lr, info in results.items():
        epochs_range = range(1, len(info["val_f1_epoch"]) + 1)
        plt.plot(epochs_range, info["val_f1_epoch"], label=f"Val F1 lr={lr}")

        epoch_opt = info["epoch_optimal"]
        f1_opt = info["val_f1_epoch"][epoch_opt-1]
        plt.scatter(epoch_opt, f1_opt, marker="o", color="red", zorder=5)
        plt.text(epoch_opt, f1_opt, f"opt={epoch_opt}", fontsize=8, color="red")

    plt.xlabel("Epochs")
    plt.ylabel("Validation F1-score")
    plt.title("Validation F1 per Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Learning Curve: Validation Accuracy ---
    plt.figure(figsize=(10,6))
    for lr, info in results.items():
        epochs_range = range(1, len(info["val_acc_epoch"]) + 1)
        plt.plot(epochs_range, info["val_acc_epoch"], label=f"Val Acc lr={lr}")

        epoch_opt = info["epoch_optimal"]
        acc_opt = info["val_acc_epoch"][epoch_opt-1]
        plt.scatter(epoch_opt, acc_opt, marker="o", color="red", zorder=5)
        plt.text(epoch_opt, acc_opt, f"opt={epoch_opt}", fontsize=8, color="red")

    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy per Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Bảng kết quả ---
    df_results = pd.DataFrame([
        {
            "Learning Rate": lr,
            "Penalty": penalty,
            "Acc_final": info["acc_final"],
            "F1_final": info["f1_final"],
            "Acc_at_opt": info["acc_at_opt_epoch"],
            "F1_at_opt": info["f1_at_opt_epoch"],
            "Epoch_optimal": info["epoch_optimal"],
            "Total time": info["time_total_epoch"],
            "Time_if_opt": info["time_opt_epoch"]
        }
        for lr, info in results.items()
    ])

    print("\n📊 Kết quả tổng hợp:")
    print(df_results.sort_values("F1_at_opt", ascending=False))

    print(f"\n✅ Best model: lr={best_lr}, penalty={penalty}, "
          f"F1_opt={best_f1:.4f}")

    return results, best_model, df_results


def accelerated_gd_run_all(X_train, y_train, X_val, y_val,
                           learning_rates,
                           epochs=1000,
                           penalty=None,
                           lam=0.01,
                           momentum=0.9,
                           tol=1e-3):
    results = {}
    best_f1 = -1
    best_lr = None
    best_model = None

    for lr in learning_rates:
        print(f"\n⚡ Training with Accelerated GD, lr={lr}, penalty={penalty}, momentum={momentum}...")

        # Train bằng accelerated gradient descent (phiên bản có lưu history_w)
        w, history_w, train_losses, times = accelerated_gd(
            X_train, y_train,
            lr=lr, epochs=epochs,
            penalty=penalty, lam=lam,
            momentum=momentum, tol=tol
        )

        # --- Evaluate trên Validation ---
        val_losses_epoch, val_acc_epoch, val_f1_epoch = [], [], []
        for w_epoch in history_w:
            val_losses_epoch.append(logistic_loss(X_val, y_val, w_epoch, penalty, lam))
            y_val_pred = predict(X_val, w_epoch)
            val_acc_epoch.append(accuracy_score(y_val, y_val_pred))
            val_f1_epoch.append(f1_score(y_val, y_val_pred, average="macro"))

        # Epoch tối ưu (theo val_loss)
        epoch_opt = int(np.argmin(val_losses_epoch))
        w_opt = history_w[epoch_opt]

        # Metrics tại epoch tối ưu
        y_val_pred_opt = predict(X_val, w_opt)
        acc_opt = accuracy_score(y_val, y_val_pred_opt)
        f1_opt = f1_score(y_val, y_val_pred_opt, average="macro")

        # Metrics tại epoch cuối
        y_val_pred_final = predict(X_val, w)
        acc_final = accuracy_score(y_val, y_val_pred_final)
        f1_final = f1_score(y_val, y_val_pred_final, average="macro")

        results[lr] = {
            "weights": w,
            "history_w": history_w,
            "train_losses": train_losses,
            "val_losses_epoch": val_losses_epoch,
            "val_acc_epoch": val_acc_epoch,
            "val_f1_epoch": val_f1_epoch,
            "times": times,
            "epoch_optimal": epoch_opt + 1,
            "acc_at_opt_epoch": acc_opt,
            "f1_at_opt_epoch": f1_opt,
            "acc_final": acc_final,
            "f1_final": f1_final,
            "time_opt_epoch": times[epoch_opt],
            "time_total_epoch": times[-1]
        }

        # Best model theo F1 tại epoch tối ưu
        if f1_opt > best_f1:
            best_f1 = f1_opt
            best_lr = lr
            best_model = results[lr]

    # --- Learning Curve: Train vs Val Loss ---
    plt.figure(figsize=(10,6))
    for lr, info in results.items():
        epochs_range = range(1, len(info["train_losses"]) + 1)
        plt.plot(epochs_range, info["train_losses"], label=f"Train lr={lr}")
        plt.plot(epochs_range, info["val_losses_epoch"], "--", label=f"Val lr={lr}")

        # Đánh dấu epoch tối ưu
        epoch_opt = info["epoch_optimal"]
        val_loss_opt = info["val_losses_epoch"][epoch_opt-1]
        plt.scatter(epoch_opt, val_loss_opt, color="red", zorder=5)
        plt.text(epoch_opt, val_loss_opt, f"opt={epoch_opt}", fontsize=8, color="red")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Learning Curve (Accelerated GD): Train vs Val Loss (penalty={penalty})")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Learning Curve: Validation F1 ---
    plt.figure(figsize=(10,6))
    for lr, info in results.items():
        epochs_range = range(1, len(info["val_f1_epoch"]) + 1)
        plt.plot(epochs_range, info["val_f1_epoch"], label=f"Val F1 lr={lr}")

        epoch_opt = info["epoch_optimal"]
        f1_opt = info["val_f1_epoch"][epoch_opt-1]
        plt.scatter(epoch_opt, f1_opt, marker="o", color="red", zorder=5)
        plt.text(epoch_opt, f1_opt, f"opt={epoch_opt}", fontsize=8, color="red")

    plt.xlabel("Epochs")
    plt.ylabel("Validation F1-score")
    plt.title("Validation F1 per Epoch (Accelerated GD)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Learning Curve: Validation Accuracy ---
    plt.figure(figsize=(10,6))
    for lr, info in results.items():
        epochs_range = range(1, len(info["val_acc_epoch"]) + 1)
        plt.plot(epochs_range, info["val_acc_epoch"], label=f"Val Acc lr={lr}")

        epoch_opt = info["epoch_optimal"]
        acc_opt = info["val_acc_epoch"][epoch_opt-1]
        plt.scatter(epoch_opt, acc_opt, marker="o", color="red", zorder=5)
        plt.text(epoch_opt, acc_opt, f"opt={epoch_opt}", fontsize=8, color="red")

    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy per Epoch (Accelerated GD)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Bảng kết quả ---
    df_results = pd.DataFrame([
        {
            "Learning Rate": lr,
            "Penalty": penalty,
            "Momentum": momentum,
            "Acc_final": info["acc_final"],
            "F1_final": info["f1_final"],
            "Acc_at_opt": info["acc_at_opt_epoch"],
            "F1_at_opt": info["f1_at_opt_epoch"],
            "Epoch_optimal": info["epoch_optimal"],
            "Total time": info["time_total_epoch"],
            "Time_if_opt": info["time_opt_epoch"]
        }
        for lr, info in results.items()
    ])

    print("\n📊 Kết quả tổng hợp:")
    print(df_results.sort_values("F1_at_opt", ascending=False))

    print(f"\n✅ Best Accelerated GD model: lr={best_lr}, penalty={penalty}, "
          f"F1_opt={best_f1:.4f}")

    return results, best_model, df_results

def newton_fixed_lr_run_all(X_train, y_train, X_test, y_test, learning_rates, epochs=50, penalty=None, lam=0.01):
    """
    Chạy Newton method với nhiều learning rate cố định.
    Vẽ loss curve theo epochs và theo thời gian, tính accuracy và chọn model tốt nhất.
    """
    results = {}
    best_acc = -1
    best_lr = None
    best_model = None

    # --- Vẽ Loss vs Epochs ---
    plt.figure(figsize=(8,6))
    for lr in learning_rates:
        start = time.time()
        w, losses = newton_method(X_train, y_train, epochs=epochs, lr=lr, penalty=penalty, lam=lam)
        end = time.time()

        # Predict & Evaluate
        y_pred = predict(X_test, w)
        acc = accuracy_score(y_test, y_pred)

        results[lr] = {"weights": w, "losses": losses, "accuracy": acc, "time": end-start}
        plt.plot(losses, label=f"lr={lr} (acc={acc:.3f})")

        # Update best model
        if acc > best_acc:
            best_acc = acc
            best_lr = lr
            best_model = {"weights": w, "losses": losses, "accuracy": acc, "lr": lr, "time": end-start}

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Newton (fixed lr) Loss vs Epochs (penalty={penalty})")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Vẽ Loss vs Time ---
    plt.figure(figsize=(8,6))
    for lr, info in results.items():
        time_axis = np.linspace(0, info["time"], len(info["losses"]))
        plt.plot(time_axis, info["losses"], label=f"lr={lr} (acc={info['accuracy']:.3f})")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Loss")
    plt.title(f"Newton (fixed lr) Loss vs Time (penalty={penalty})")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"\nBest learning rate: {best_lr}, Accuracy: {best_acc:.4f}, Time: {best_model['time']:.2f} sec")

    return results, best_model

def newton_backtracking_run(X_train, y_train, X_test, y_test, epochs=50, penalty=None, lam=0.01, alpha=0.25, beta=0.8):
    """
    Chạy Newton method với backtracking line search.
    Vẽ loss curve theo epochs và theo thời gian, tính accuracy.
    """
    start = time.time()
    w, losses = newton_backtracking(X_train, y_train, epochs=epochs, penalty=penalty, lam=lam, alpha=alpha, beta=beta)
    end = time.time()

    # Predict & Evaluate
    y_pred = predict(X_test, w)
    acc = accuracy_score(y_test, y_pred)

    # --- Vẽ Loss vs Epochs ---
    plt.figure(figsize=(8,6))
    plt.plot(losses, label=f"Backtracking (acc={acc:.3f})")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Newton (backtracking) Loss vs Epochs (penalty={penalty})")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Vẽ Loss vs Time ---
    plt.figure(figsize=(8,6))
    time_axis = np.linspace(0, end-start, len(losses))
    plt.plot(time_axis, losses, label=f"Backtracking (acc={acc:.3f})")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Loss")
    plt.title(f"Newton (backtracking) Loss vs Time (penalty={penalty})")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"\nNewton (backtracking) → Accuracy: {acc:.4f}, Time: {end-start:.2f} sec")

    return {"weights": w, "losses": losses, "accuracy": acc, "time": end-start}
