import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_recall_curve, average_precision_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('https://raw.githubusercontent.com/tigasparzin/Machine-Learning/refs/heads/main/data/StressExp.csv')

if 'Timestamp' in df.columns:
    df = df.drop(columns=['Timestamp'])

X = df[['AcademicStage','PeerPressure','HomePressure','StudyEnv','Strategy','BadHabits','AcademicComp']].copy()

# Binarização: alto >=3 -> 0 ; baixo <3 -> 1
y = (df['Stress'] >= 3).map({True: 0, False: 1})

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Imputação mínima
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

## Modelo 
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=5,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)
y_pred  = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 0]  # prob da classe 0 = alto

print(f"Accuracy: {(y_pred == y_test).mean():.2f}")
print()

labels = [0, 1]  # 0=alto, 1=baixo
cm = confusion_matrix(y_test, y_pred, labels=labels)
fig, ax = plt.subplots(figsize=(7, 5))
ConfusionMatrixDisplay(cm, display_labels=['alto(0)','baixo(1)']).plot(
    ax=ax, cmap=plt.cm.Blues, values_format="d", colorbar=False
)
ax.set_title("Matriz de Confusão (contagens) - RF binário")
ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
buf = StringIO(); plt.savefig(buf, format="svg", transparent=True, bbox_inches="tight")
print(buf.getvalue()); plt.close(fig)


ap = average_precision_score((y_test==0).astype(int), y_proba)
print(f"Average Precision (classe alto=0): {ap:.2f}")
print()
prec, rec, thr = precision_recall_curve((y_test==0).astype(int), y_proba)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(rec, prec)
ax.set_title("Precision-Recall (alto=0)"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
buf = StringIO(); plt.savefig(buf, format="svg", transparent=True, bbox_inches="tight")
print(buf.getvalue()); plt.close(fig)


feat_names = X_train.columns.to_numpy()
importances = clf.feature_importances_
top_idx = np.argsort(importances)[::-1][:10]
print("Top 10 features:")
for i in top_idx:
    print(f"{feat_names[i]}: {importances[i]:.2f}")
    print()
