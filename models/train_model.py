import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/student_data.csv")

X = df[[
    "total_marks", "total_absences", "supplementary_exams",
    "family_support", "extra_curricular", "wants_higher_education", "internet_access"
]]
y = df["Risk_Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Feature importance plot
feature_imp = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(8,5))
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("models/feature_importance.png")
plt.show()

# Save model
joblib.dump(model, "models/student_risk_model.pkl")
print("Model saved to models/student_risk_model.pkl")