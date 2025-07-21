import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageTk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Save files to Desktop
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

# Global objects
analyzer = SentimentIntensityAnalyzer()
df = None
vectorizer = None
model = None
text_col = None
label_col = None

# ---------------------- Clean Text ----------------------
def clean_text(text):
    try:
        if pd.isna(text) or not isinstance(text, str):
            return None
        text = re.sub(r"http\S+|@\w+|#\w+|[^A-Za-z\s]", "", text)
        text = text.lower().strip()
        if not text or text.isspace():
            return None
        return text
    except:
        return None

# ---------------------- Detect Columns ----------------------
def detect_columns(data):
    global text_col, label_col
    text_col, label_col = None, None
    for col in data.columns:
        if data[col].dtype == object and 'text' in col.lower():
            text_col = col
        if 'label' in col.lower() or 'sentiment' in col.lower():
            label_col = col
    if not text_col or not label_col:
        raise Exception("Text or label column not found!")

def map_labels(series):
    mapping = {-1: 'negative', 0: 'neutral', 1: 'positive'}
    if series.dtype in [int, float]:
        return series.map(mapping).fillna(series)
    return series

# ---------------------- Heuristic Sentiment ----------------------
def analyze_custom_sentiment(text):
    if not text or not isinstance(text, str) or len(text.split()) == 0:
        return 'invalid'
    if any(word in text for word in ['not', 'never']) and any(pos in text for pos in ['great', 'amazing', 'fantastic']):
        return 'sarcastic-negative'
    for keyword in ['but', 'however', 'although']:
        if keyword in text:
            parts = text.split(keyword)
            if len(parts) >= 2:
                vec = vectorizer.transform([parts[1]])
                return model.predict(vec)[0]
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    if pred == 'neutral':
        score = analyzer.polarity_scores(text)['compound']
        if score >= 0.4:
            return 'positive'
        elif score <= -0.4:
            return 'negative'
    return pred

# ---------------------- Train Model ----------------------
def train_model():
    global model, vectorizer, df
    try:
        detect_columns(df)
        df.dropna(subset=[text_col, label_col], inplace=True)
        df[text_col] = df[text_col].astype(str).apply(clean_text)
        df = df[df[text_col].notna()]
        df[label_col] = map_labels(df[label_col])
        most_freq = df[label_col].mode()[0]
        untrained_accuracy = (df[label_col] == most_freq).mean()
        vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_df=0.95, stop_words='english')
        X = vectorizer.fit_transform(df[text_col])
        y = df[label_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
        model = LinearSVC(class_weight='balanced')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        trained_accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        output_text.delete('1.0', tk.END)
        output_text.insert(tk.END, f"✅ Model Trained Successfully!\n\nAccuracy: {trained_accuracy:.2f}\n\n{report}")

        # Save charts
        cm = confusion_matrix(y_test, y_pred, labels=['negative', 'neutral', 'positive'])
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['negative', 'neutral', 'positive'],
                    yticklabels=['negative', 'neutral', 'positive'])
        plt.title("Confusion Matrix")
        plt.tight_layout()
        cm_path = os.path.join(desktop_path, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()

        plt.figure(figsize=(5, 4))
        bars = plt.bar(['Untrained', 'Trained'], [untrained_accuracy, trained_accuracy],
                       color=['#ff6f61', '#4caf50'])
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{bar.get_height():.2f}", ha='center', fontweight='bold')
        plt.ylim(0, 1)
        plt.title("Model Accuracy")
        plt.tight_layout()
        acc_path = os.path.join(desktop_path, "accuracy_chart.png")
        plt.savefig(acc_path)
        plt.close()

        show_image_popup(cm_path, acc_path)

    except Exception as e:
        messagebox.showerror("Training Error", str(e))

# ---------------------- Image Popup ----------------------
def show_image_popup(cm_path, acc_path):
    popup = tk.Toplevel(root)
    popup.title("📊 Model Evaluation")
    popup.geometry("700x320")
    popup.configure(bg="#f9f9f9")
    try:
        img1 = Image.open(cm_path).resize((320, 280))
        img2 = Image.open(acc_path).resize((320, 280))
        tk_img1 = ImageTk.PhotoImage(img1)
        tk_img2 = ImageTk.PhotoImage(img2)
        tk.Label(popup, image=tk_img1, bg="#f9f9f9").grid(row=0, column=0, padx=10)
        tk.Label(popup, image=tk_img2, bg="#f9f9f9").grid(row=0, column=1, padx=10)
        popup.mainloop()
    except Exception as e:
        print("Image popup error:", e)

# ---------------------- Predict Input ----------------------
def predict_input():
    if model is None:
        messagebox.showwarning("Train First", "Train the model first.")
        return
    raw_text = text_input.get()
    cleaned = clean_text(raw_text)
    if not cleaned:
        messagebox.showerror("Invalid", "Enter valid text (not empty or symbols).")
        return
    pred = analyze_custom_sentiment(cleaned)
    if pred == 'invalid':
        messagebox.showerror("Invalid Input", "Provide meaningful English text.")
    else:
        output_text.insert(tk.END, f"\n📝 Input: {raw_text}\n🔍 Prediction: {pred}\n")

# ---------------------- Predict File ----------------------
def predict_file():
    if model is None:
        messagebox.showwarning("Train First", "Train the model first.")
        return
    path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not path:
        return
    try:
        pred_df = pd.read_csv(path)
        col = pred_df.columns[0]
        pred_df[col] = pred_df[col].astype(str).apply(clean_text)
        pred_df = pred_df[pred_df[col].notna()]
        results = [analyze_custom_sentiment(row) for row in pred_df[col]]
        pred_df = pd.DataFrame({'Text': pred_df[col], 'Predicted': results})
        output_text.insert(tk.END, "\n📁 File Predictions:\n")
        for i, row in pred_df.iterrows():
            output_text.insert(tk.END, f"{row['Text']} → {row['Predicted']}\n")
        save_path = os.path.join(desktop_path, "Prediction Output File.csv")
        pred_df.to_csv(save_path, index=False)
        messagebox.showinfo("Saved", f"📄 Predictions saved at:\n{save_path}")
    except Exception as e:
        messagebox.showerror("Prediction Error", str(e))

# ---------------------- Upload Dataset ----------------------
def upload_file():
    global df
    path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if path:
        df = pd.read_csv(path)
        messagebox.showinfo("Loaded", "Dataset uploaded successfully!")

# ---------------------- Reset App ----------------------
def reset_app():
    global df, model, vectorizer, text_col, label_col
    df = None
    model = None
    vectorizer = None
    text_col = None
    label_col = None
    text_input.delete(0, tk.END)
    output_text.delete('1.0', tk.END)
    messagebox.showinfo("Reset", "App reset successful.")

# ---------------------- GUI Setup ----------------------
root = tk.Tk()
root.title("🎯 Sentiment Analyzer")
root.geometry("1000x700")
root.configure(bg="#eef2f3")

tk.Label(root, text="🧠 Sentiment Analysis Tool", font=("Arial", 18, "bold"),
         bg="#eef2f3", fg="#1f4068").pack(pady=10)

btn_frame = tk.Frame(root, bg="#eef2f3")
btn_frame.pack(pady=10)

ttk.Button(btn_frame, text="📂 Upload Dataset", command=upload_file).grid(row=0, column=0, padx=5)
ttk.Button(btn_frame, text="⚙️ Train Model", command=train_model).grid(row=0, column=1, padx=5)
ttk.Button(btn_frame, text="📁 Predict File", command=predict_file).grid(row=0, column=2, padx=5)
ttk.Button(btn_frame, text="🔁 Reset", command=reset_app).grid(row=0, column=3, padx=5)

tk.Label(root, text="📝 Type Sentence:", bg="#eef2f3", font=("Arial", 12)).pack()
text_input = tk.Entry(root, width=90, font=("Arial", 11))
text_input.pack(pady=5)
ttk.Button(root, text="🔍 Predict Sentiment", command=predict_input).pack()

tk.Label(root, text="📊 Output", font=("Arial", 13, "bold"), bg="#eef2f3").pack()
output_text = tk.Text(root, height=15, width=110, font=("Courier", 10))
output_text.pack(pady=10)

root.mainloop()
