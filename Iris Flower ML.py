from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tkinter as tk
from tkinter import messagebox

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0:'setosa', 1:'versicolor', 2:'virginica'})


X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "KNN": KNeighborsClassifier(n_neighbors=3)
}

print("🔹 Model Accuracy Comparison:\n")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name}: {accuracy_score(y_test, y_pred):.2f}")


dt_model = models["Decision Tree"]
plt.figure(figsize=(12,6))
plot_tree(dt_model, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("Decision Tree Visualization")
plt.show()


joblib.dump(dt_model, "iris_model.pkl")
print("✅ Model saved as iris_model.pkl")

def predict_species():
    try:
        features = [float(e1.get()), float(e2.get()), float(e3.get()), float(e4.get())]
        model = joblib.load("iris_model.pkl")
        prediction = model.predict([features])[0]
        messagebox.showinfo("Prediction Result", f"The predicted species is: {prediction}")
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numeric values!")

window = tk.Tk()
window.title("🌸 Iris Flower Predictor")
window.geometry("350x250")

tk.Label(window, text="Sepal Length (cm):").pack()
e1 = tk.Entry(window); e1.pack()

tk.Label(window, text="Sepal Width (cm):").pack()
e2 = tk.Entry(window); e2.pack()

tk.Label(window, text="Petal Length (cm):").pack()
e3 = tk.Entry(window); e3.pack()

tk.Label(window, text="Petal Width (cm):").pack()
e4 = tk.Entry(window); e4.pack()

tk.Button(window, text="Predict", command=predict_species, bg="lightgreen").pack(pady=10)

window.mainloop()
