
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('https://raw.githubusercontent.com/tigasparzin/Machine-Learning/refs/heads/main/data/StressExp.csv')

def preprocess(df: pd.DataFrame):
    if 'Timestamp' in df.columns:
        df = df.drop(columns=['Timestamp'])

    # define X e y
    X = df[['AcademicStage','PeerPressure','HomePressure','StudyEnv','Strategy','BadHabits','AcademicComp']].copy()
    y = df['Stress'].copy()

    # split estratificado para manter proporções das 5 classes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # imputação: StudyEnv com a moda do treino (evita leakage)
    if X_train['StudyEnv'].isna().any() or X_test['StudyEnv'].isna().any():
        mode_env = X_train['StudyEnv'].mode().iloc[0]
        X_train['StudyEnv'] = X_train['StudyEnv'].fillna(mode_env)
        X_test['StudyEnv'] = X_test['StudyEnv'].fillna(mode_env)

    # codificação categórica via LabelEncoder (mantendo o espírito do original)
    cat_cols = ['AcademicStage','StudyEnv','Strategy','BadHabits']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))

        # garantir que valores desconhecidos no teste sejam mapeados com segurança
        X_test[col] = X_test[col].astype(str).map(lambda v: v if v in le.classes_ else le.classes_[0])
        # alinhar classes
        import numpy as _np
        le.classes_ = _np.unique(_np.concatenate([le.classes_, X_test[col].unique()]))
        X_test[col] = le.transform(X_test[col])
        encoders[col] = le

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess(df)

# Modelo: Random Forest (multiclasse)
clf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Matriz de confusão com 5 classes (labels ordenadas)
labels = np.sort(df['Stress'].unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

# plot
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax, cmap=plt.cm.Blues, values_format="d", colorbar=False)

ax.set_title("Matriz de Confusão - Random Forest")
ax.set_xlabel("Previsto")
ax.set_ylabel("Real")

# exportar para SVG e imprimir como no original
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True, bbox_inches="tight")
svg_text = buffer.getvalue()
print(svg_text)

plt.close(fig)
