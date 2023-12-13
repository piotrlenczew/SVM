from svm import SVM, SVMParams
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
wine_quality = fetch_ucirepo(id=186)

X = wine_quality.data.features
y = wine_quality.data.targets

y_binary = np.where(y > 5, 1, -1)

limit = 4000
X = X[:limit]
y_binary = y_binary[:limit]

X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

y_train = y_train.ravel()
y_test = y_test.ravel()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = 1e-3

svm_params = SVMParams(C=100, kernel='rbf', sigma=1)
svm = SVM(svm_params)
losses = svm.fit(X_train_scaled, y_train, lr)
plt.plot(losses)
plt.title(f"loss per epochs for lr={lr}")
plt.show()
print("Train score", svm.score(X_test_scaled, y_test))
