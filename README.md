Great! This second notebook is a **classification lab using Decision Trees** on a medical dataset called **`drug200.csv`**, which contains patient information and the type of drug prescribed to them.

---

## 🔍 Detailed Explanation of the Notebook

---

### ✅ **Objective**
To build a **Decision Tree Classifier** that predicts the **drug** to prescribe to a patient based on features like age, sex, blood pressure, cholesterol, and sodium/potassium levels.

---

### 📦 **1. Package Installation**
```python
!pip install numpy==2.2.0
!pip install pandas==2.2.3
!pip install scikit-learn==1.6.0
!pip install matplotlib==3.9.3
```
These ensure compatible versions of important libraries are used:
- `NumPy`, `Pandas`: Data manipulation
- `Matplotlib`: Plotting
- `scikit-learn`: Machine learning models

---

### 📥 **2. Import Libraries**
```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
```
It imports tools for:
- **Preprocessing**: Label encoding categorical variables
- **Modeling**: Training a decision tree
- **Evaluation**: Accuracy, confusion matrix, etc.
- **Visualization**: Drawing the tree

---

### 📊 **3. Load Dataset**
```python
my_data = pd.read_csv('drug200.csv')
```
Dataset contains:
- **Features**: Age, Sex, Blood Pressure, Cholesterol, Na_to_K (sodium to potassium ratio)
- **Target**: Drug (DrugA, DrugB, ..., DrugY)

---

### 🔍 **4. Data Inspection**
```python
my_data.info()
```
Gives insight into:
- Number of records
- Data types
- Non-null values

---

### 🔧 **5. Preprocessing**
```python
label_encoder = LabelEncoder()
my_data['Sex'] = label_encoder.fit_transform(my_data['Sex'])
my_data['BP'] = label_encoder.fit_transform(my_data['BP'])
my_data['Cholesterol'] = label_encoder.fit_transform(my_data['Cholesterol'])
```
Transforms categorical features into numbers so they can be used in the decision tree:
- `Sex`: Male/Female → 1/0
- `BP`: HIGH/NORMAL/LOW → e.g. 2/1/0
- `Cholesterol`: HIGH/NORMAL → e.g. 1/0

---

Here’s a **continued and detailed breakdown** of the notebook, step-by-step:

---

### 🧼 **6. Missing Data Check**
```python
my_data.isnull().sum()
```
This checks for **missing values** in each column. All values are expected to be complete.

---

### 🔢 **7. Convert Target to Numeric**
```python
custom_map = {'drugA': 0, 'drugB': 1, 'drugC': 2, 'drugX': 3, 'drugY': 4}
my_data['Drug_num'] = my_data['Drug'].map(custom_map)
```
- Converts the `Drug` column (target class) into numeric values.
- Makes it easier to work with classification models.

---

### 🧠 **8. Dataset Info After Encoding**
```python
my_data.info()
```
Now the data types show encoded categorical columns and the numeric target column (`Drug_num`).

---

### 📉 **9. Histograms of All Features**
```python
my_data.hist(bins=30, color='r', figsize=(16, 16))
```
This visualizes **distributions of numeric features** like Age and Na_to_K, which helps understand their spread.

---

### 📈 **10. Feature Correlation**
```python
data = my_data.select_dtypes(include='number')
corr = data.corr()
```
This selects only numerical columns to compute the **correlation matrix**, which shows how features are related to each other.

---

### 🔥 **11. Heatmap of Correlations**
```python
import seaborn as sns
plt.figure(figsize=(8, 8))
sns.heatmap(corr, annot=True)
plt.show()
```
- Visual tool showing how strongly features are correlated.
- Helpful for identifying redundant or predictive features.

---

### 📊 **12. Drug Class Distribution (Twice)**
```python
drugs = my_data['Drug'].value_counts()
plt.bar(drugs.index, drugs.values, color='r')
```
and
```python
category_counts = my_data['Drug'].value_counts()
plt.bar(category_counts.index, category_counts.values, color='blue')
```
- These two cells plot how often each drug label occurs in the dataset.
- Confirms class balance or imbalance (important for training accuracy).

---

Now let’s dive into the **core modeling part** of the notebook: building and evaluating the Decision Tree classifier.

---

## 🌲 **13. Define Features & Target**
```python
y = my_data['Drug']
X = my_data.drop(['Drug', 'Drug_num'], axis=1)
```
- `X`: All features like Age, Sex, BP, Cholesterol, Na_to_K
- `y`: The actual drug class (string labels)

---

## 🧪 **14. Split Dataset**
```python
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=32)
```
- 70% for training, 30% for testing
- `random_state=32` ensures reproducibility

---

## 🧠 **15. Train a Decision Tree**
```python
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
drugTree.fit(X_trainset, y_trainset)
```
- Uses **Entropy** as the splitting criterion (information gain)
- `max_depth=4` limits the depth to prevent overfitting

---

## 📊 **16. Predict on Test Set**
```python
tree_predictions = drugTree.predict(X_testset)
```
Makes predictions for unseen data (testing set).

---

## 🎯 **17. Accuracy Evaluation**
```python
print("Decision Trees's Accuracy: ", metrics.accuracy_score(y_testset, tree_predictions))
```
Compares predicted vs. actual drug labels and prints the model's **accuracy**.

---

## 🌳 **18. Visualize the Tree**
```python
plot_tree(drugTree)
plt.show()
```
- Displays the actual decision tree, showing:
  - Feature splits
  - Threshold values
  - Class outcomes

---

## 🧪 **19. Another Tree with Different Depth**
```python
tree_model = DecisionTreeClassifier(criterion='entropy', max_depth=3)
tree_model.fit(X_trainset, y_trainset)
```
- A second tree is built with `max_depth=3` to compare model behavior with a shallower tree.

---

## ✅ **20. Accuracy of Second Tree**
```python
pred = tree_model.predict(X_testset)
print(f"Accuracy = {np.round(100*metrics.accuracy_score(y_testset, pred), 2)}%")
```
- Calculates accuracy for the second tree
- Displays in percentage format

---

## 🌿 **21. Visualize Second Tree**
```python
plot_tree(tree_model)
plt.show()
```
- Shows the structure of the second (shallower) tree.

---

### ✅ Summary of Model Section:

| Step | Description |
|------|-------------|
| Model | Decision Tree Classifier |
| Criterion | Entropy (Information Gain) |
| Output | Drug prescription (DrugA–Y) |
| Accuracy | Evaluated on test set |
| Visualization | Tree structure shown using `plot_tree()` |

---
