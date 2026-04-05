import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data
df = pd.read_csv('../data.csv') # <-- Change to your uploaded filename!

# 🌟 THE FIX: Remove any rows where 'ClassLabel' is empty (NaN)
df = df.dropna(subset=['ClassLabel'])

# 2. Prepare Features (X) and Target (y)
X = df.drop(['URL', 'ClassLabel'], axis=1)
y = df['ClassLabel']

# 3. Split the data (80% Training, 20% Testing) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Initialize and Train the Decision Tree
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 5. Make Predictions
y_pred = model.predict(X_test)

# 6. Evaluate the Results
print(df.shape)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print(X.shape)

# 7. Visual Heatmap (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title('Phishing Detection Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()
