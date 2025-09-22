import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# carregar os dados
df = pd.read_csv('https://raw.githubusercontent.com/tigasparzin/Machine-Learning/refs/heads/main/data/StressExp.csv')

# pré-processamento (label encoding das categóricas)
categorical_cols = ['AcademicStage', 'StudyEnv', 'Strategy', 'BadHabits']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# separar features e target
X = df.drop(columns=['Timestamp', 'Stress'])
y = df['Stress'].astype(int)

# dividir em treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# treinar KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

# gerar matriz de confusão
labels = np.sort(np.unique(np.concatenate([y_test, predictions])))
cm = confusion_matrix(y_test, predictions, labels=labels)

# plotar matriz de confusão
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax, cmap=plt.cm.Blues, values_format="d", colorbar=False)

ax.set_title("Matriz de Confusão - KNN")
ax.set_xlabel("Previsto")
ax.set_ylabel("Real")

# exportar para SVG
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True, bbox_inches="tight")
svg_text = buffer.getvalue()
print(svg_text)

plt.close(fig)