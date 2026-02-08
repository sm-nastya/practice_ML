from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from parse_dataset import GetProcessedDF
import pandas as pd

df_obj = GetProcessedDF("hh.csv")
hh_df = df_obj.get_dataframe()
y = hh_df["Уровень кандидата"]
X = hh_df.drop(columns=["Уровень кандидата"])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
