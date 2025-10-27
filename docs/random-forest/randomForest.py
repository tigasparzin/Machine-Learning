import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Carregar base
df = pd.read_csv("https://raw.githubusercontent.com/tigasparzin/Machine-Learning/refs/heads/main/data/StressExp.csv")

y = (df["Stress"])

# Features fixas da base
X = df.drop(columns=["Timestamp", "Stress"])
X = pd.get_dummies(
    X,
    columns=["AcademicStage", "StudyEnv", "Strategy", "BadHabits"],
    drop_first=False
)

# Split estratificado
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    max_features='sqrt',
    random_state=42
)
rf.fit(X_train, y_train)

# Avaliação
pred = rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, pred):.2f}")
print()

# Importâncias (vetor)
print("Feature Importances:", rf.feature_importances_)
