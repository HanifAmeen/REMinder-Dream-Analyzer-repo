import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
import os

# --- Step 2A: Load Dataset ---
dataset_path = r"C:\Users\amjad\Downloads\Research Papers 2025\Dream Journal\Datasets\dreambank.csv"
df = pd.read_csv(dataset_path)

# Inspect columns
print(df.columns)

# --- Step 2B: Prepare Data ---
# Keep only the relevant columns
df = df[['report', 'emotion']].dropna()

# Encode target labels
le = LabelEncoder()
df['emotion_encoded'] = le.fit_transform(df['emotion'])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['report'], df['emotion_encoded'], test_size=0.2, random_state=42
)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Data preparation complete!")
print(f"Number of training samples: {X_train_vec.shape[0]}")
print(f"Number of features: {X_train_vec.shape[1]}")

# --- Step 3: Train Model ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# --- Step 4: Evaluate Model ---
y_pred = model.predict(X_test_vec)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# --- Step 5: Save Model and Vectorizer ---
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, os.path.join(model_dir, "dream_emotion_model.pkl"))
joblib.dump(vectorizer, os.path.join(model_dir, "tfidf_vectorizer.pkl"))
joblib.dump(le, os.path.join(model_dir, "label_encoder.pkl"))

print("Model, vectorizer, and label encoder saved successfully!")
