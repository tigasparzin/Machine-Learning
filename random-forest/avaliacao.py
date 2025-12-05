import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('https://raw.githubusercontent.com/tigasparzin/Machine-Learning/refs/heads/main/data/StressExp.csv')

if 'Timestamp' in df.columns:
    df = df.drop(columns=['Timestamp'])

X = df[['AcademicStage','PeerPressure','HomePressure','StudyEnv','Strategy','BadHabits','AcademicComp']].copy()
y = df['Stress'].astype(int)  # sem binarização (1..5)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

if X_train['StudyEnv'].isna().any() or X_test['StudyEnv'].isna().any():
    mode_env = X_train['StudyEnv'].mode().iloc[0]
    X_train['StudyEnv'] = X_train['StudyEnv'].fillna(mode_env)
    X_test['StudyEnv']  = X_test['StudyEnv'].fillna(mode_env)

cat_cols = ['AcademicStage','StudyEnv','Strategy','BadHabits']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = X_test[col].astype(str).where(X_test[col].astype(str).isin(le.classes_), le.classes_[0])
    import numpy as _np
    le.classes_ = _np.unique(_np.concatenate([le.classes_, X_test[col].unique()]))
    X_test[col] = le.transform(X_test[col])
    encoders[col] = le

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=5,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"Accuracy: {(y_pred == y_test).mean():.2f}\n")
print(classification_report(y_test, y_pred, digits=3))

labels = np.sort(df['Stress'].unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
fig, ax = plt.subplots(figsize=(7, 5))
ConfusionMatrixDisplay(cm, display_labels=labels).plot(
    ax=ax, cmap=plt.cm.Blues, values_format="d", colorbar=False
)
ax.set_title("Matriz de Confusão - RF (5 classes)")
ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
buf = StringIO(); plt.savefig(buf, format="svg", transparent=True, bbox_inches="tight")
print(buf.getvalue()); plt.close(fig)

feat_names = X_train.columns.to_numpy()
importances = clf.feature_importances_
top_idx = np.argsort(importances)[::-1][:10]
print("Top 10 features:")
for i in top_idx:
    print(f"{feat_names[i]}: {importances[i]:.3f}")
