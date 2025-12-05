import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
from io import StringIO

# Carrega a base
df = pd.read_csv('https://raw.githubusercontent.com/tigasparzin/Machine-Learning/refs/heads/main/data/StressExp.csv')

# Binarização
y = (df['Stress'] >= 4).map({False: 1, True: 0})

# Usa duas variáveis numéricas como features
X = df[["PeerPressure", "HomePressure"]].values

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6))

kernels = {
    'linear': ax1,
    'sigmoid': ax2,
    'poly': ax3,
    'rbf': ax4
}

for k, ax in kernels.items():
    svm = SVC(kernel=k, C=1)
    svm.fit(X, y)

    DecisionBoundaryDisplay.from_estimator(
        svm,
        X,
        response_method="predict",
        alpha=0.8,
        cmap="Pastel1",
        ax=ax
    )

    ax.scatter(
        X[:, 0], X[:, 1],
        c=y,
        s=20, edgecolors="black"
    )

    ax.set_title(k)
    ax.set_xticks([])
    ax.set_yticks([])

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
plt.close()
