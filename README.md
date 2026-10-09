# 🥛 Milk Quality Prediction Using Machine Learning

A machine learning project that predicts milk quality based on its physical and sensory characteristics. This project explores data cleaning, exploratory data analysis, feature normalization, model training, and evaluation using **K-Nearest Neighbors (KNN)** and **Decision Tree Classification**.

## 📌 Table of Contents

* [Overview](#-overview)
* [Project Objectives](#-project-objectives)
* [Technologies and Libraries](#-technologies-and-libraries)
* [Dataset Description](#-dataset-description)
* [Project Workflow](#-project-workflow)
* [Machine Learning Models](#-machine-learning-models)
* [Model Evaluation](#-model-evaluation)
* [Installation and Setup](#-installation-and-setup)
* [How to Run](#-how-to-run)
* [Project Structure](#-project-structure)
* [Future Improvements](#-future-improvements)
* [Author](#-author)
* [License](#-license)

## 🔍 Overview

Milk quality assessment is an important task in the dairy industry. Milk can be classified into different quality categories based on characteristics such as pH, temperature, taste, odor, fat content, turbidity, and color.

The goal of this project is to develop and compare supervised machine learning classifiers that predict milk quality using these input features.

The project follows a typical machine learning workflow, starting with loading and exploring the dataset and ending with model evaluation and visualization of a trained decision tree.

**Target variable:** `Grade`

The target is encoded into three numerical classes:

| Class          | Encoded value |
| -------------- | ------------: |
| Low quality    |             0 |
| Medium quality |             1 |
| High quality   |             2 |

## 🎯 Project Objectives

The main objectives are to:

* Load and inspect the milk quality dataset using Pandas.
* Clean and transform the target labels into numerical values.
* Explore feature distributions and class frequencies.
* Visualize selected numerical features.
* Prepare the feature matrix and target vector.
* Normalize the input features.
* Split the dataset into training and testing sets.
* Train a K-Nearest Neighbors classifier.
* Evaluate different values of K to investigate their effect on accuracy.
* Train a Decision Tree classifier and tune its maximum depth using cross-validation.
* Compare training and testing accuracy.
* Visualize the trained Decision Tree.

## 🛠️ Technologies and Libraries

The project is implemented in Python using Jupyter Notebook.

| Technology       | Purpose                                                        |
| ---------------- | -------------------------------------------------------------- |
| Python           | Core programming language                                      |
| Jupyter Notebook | Interactive development and experimentation                    |
| NumPy            | Numerical operations and array manipulation                    |
| Pandas           | Dataset loading and data manipulation                          |
| Matplotlib       | Data visualization and plotting                                |
| Scikit-learn     | Preprocessing, classification, model selection, and evaluation |
| pydotplus        | Converting the decision tree representation into a graph       |
| Graphviz         | Rendering the decision tree visualization                      |

## 📊 Dataset Description

The notebook loads the dataset from a CSV file named `data.csv`.

Each row represents a milk sample, with seven input features and one target variable.

| Feature      | Description                                   |
| ------------ | --------------------------------------------- |
| `pH`         | pH value of the milk                          |
| `Temprature` | Temperature of the milk                       |
| `Taste`      | Taste indicator: 0 for bad and 1 for good     |
| `Odor`       | Odor indicator: 0 for bad and 1 for good      |
| `Fat`        | Fat indicator: 0 for low and 1 for high       |
| `Turbidity`  | Turbidity indicator: 0 for low and 1 for high |
| `Color`      | Color measurement                             |
| `Grade`      | Target class: low, medium, or high quality    |

The feature names above follow the notebook's column naming, including `Temprature`.

### Target Class Distribution

The notebook reports the following class counts:

| Milk quality | Number of samples |
| ------------ | ----------------: |
| Low          |               429 |
| Medium       |               374 |
| High         |               256 |
| **Total**    |         **1,059** |

These counts provide an initial view of the distribution of the target classes.

> **Dataset requirement:** The notebook expects `data.csv` to be available in the working directory. Make sure the CSV file is included in your repository if its license and source permit redistribution.

## 🔄 Project Workflow

### 1. Data Loading

The dataset is loaded into a Pandas DataFrame using `pd.read_csv()`. The first rows are inspected to understand its structure and columns.

### 2. Data Cleaning and Label Encoding

The categorical `Grade` values are converted into numerical labels:

* `low` → `0`
* `medium` → `1`
* `high` → `2`

This transformation allows the target labels to be used by the classification models.

### 3. Exploratory Data Analysis

The notebook explores the dataset using:

* Descriptive statistics with `data.describe()`.
* Class frequency counts with `value_counts()`.
* Histograms of the pH and temperature columns.

These steps help examine the dataset's numerical characteristics and class distribution.

### 4. Feature and Target Preparation

The first seven columns are selected as the feature matrix, `X`, while `Grade` is used as the target, `y`.

* **Features (`X`):** pH, temperature, taste, odor, fat, turbidity, and color.
* **Target (`y`):** Encoded milk quality grade.

### 5. Feature Normalization

The notebook applies Scikit-learn's `Normalizer` to the feature matrix.

Normalization scales each sample vector to unit norm. This is distinct from standardization, which transforms features to have a mean of zero and a standard deviation of one.

### 6. Train-Test Split

The dataset is divided into training and testing subsets using `train_test_split()`.

* Training set: 80%
* Testing set: 20%
* Random state: `4`

The training set is used to fit the models, while the testing set is used to evaluate their predictions.

### 7. Model Training and Evaluation

Two classification algorithms are investigated:

* K-Nearest Neighbors (KNN)
* Decision Tree Classifier

Their training and testing accuracy scores are calculated using Scikit-learn's `accuracy_score()`.

## 🤖 Machine Learning Models

### 1. K-Nearest Neighbors (KNN)

KNN classifies a sample based on the labels of its nearest neighbors in the feature space.

In this project:

* The initial model uses `K = 3`.
* Values of K from 1 to 9 are evaluated.
* Test accuracy is calculated for each value of K.
* An accuracy plot is generated to compare the results.

The initial KNN model is configured as follows:

```python
from sklearn.neighbors import KNeighborsClassifier

k = 3
kcls = KNeighborsClassifier(n_neighbors=k)
kcls.fit(train_x, train_y)
```

The experiment with different K values helps investigate how the number of neighbors affects classification accuracy.

### 2. Decision Tree Classifier

A Decision Tree learns a sequence of feature-based decision rules to classify milk samples.

In this project, `GridSearchCV` is used to investigate the effect of different maximum tree depths. The search evaluates depths from 1 through 19 using 10-fold cross-validation and accuracy scoring.

The notebook reports a best maximum depth of `9` in its saved output.

The final classifier is configured with entropy as the splitting criterion:

```python
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=9
)

dtc.fit(train_x, train_y)
```

The trained tree is also exported using Scikit-learn's tree visualization utilities and rendered as a PNG image.

## 📈 Model Evaluation

The primary evaluation metric used in this project is **classification accuracy**.

Accuracy measures the proportion of predictions that match the actual target labels:

$$
\text{Accuracy} =
\frac{\text{Correct Predictions}}
{\text{Total Predictions}}
$$

The notebook evaluates each model on both the training and testing datasets.

### Recorded Results

The saved notebook outputs contain the following accuracy scores:

| Model                           | Training accuracy | Testing accuracy |
| ------------------------------- | ----------------: | ---------------: |
| KNN (`K = 3`)                   |            99.29% |             100% |
| Decision Tree (`max_depth = 9`) |            99.29% |             100% |

These values reflect the results stored in the notebook and should be interpreted in the context of its current preprocessing and evaluation procedure.

**Important evaluation note:** The notebook normalizes the complete feature matrix before splitting it into training and testing sets. For a more reliable estimate of performance on unseen data, fit preprocessing only on the training data and apply it to the test data through a Scikit-learn `Pipeline`. The recorded 100% test accuracy should also be validated with repeated or stratified evaluation and additional metrics before drawing conclusions about real-world performance.

## ⚙️ Installation and Setup

### Prerequisites

Make sure you have:

* Python 3.9 or a compatible Python environment.
* Jupyter Notebook or JupyterLab.
* The `data.csv` dataset file.

### 1. Clone the Repository

```bash
git clone https://github.com/sahandsbt/milk-quality-prediction.git
cd milk-quality-prediction
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install the Required Libraries

```bash
python -m pip install numpy pandas matplotlib scikit-learn jupyter pydotplus
```

To render the decision tree as a PNG image, install **Graphviz** on your operating system as well. Installing the Python package alone may not provide the Graphviz system executable.

### 4. Verify the Dataset

Ensure that `data.csv` is located in the directory from which you will run the notebook, or update the CSV path in the notebook accordingly.

## ▶️ How to Run

1. Clone the repository and install the dependencies.

2. Place `data.csv` in the appropriate directory.

3. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

4. Open `main.ipynb`.

5. Run the notebook cells sequentially, from top to bottom.

6. Review the dataset summaries, histograms, KNN accuracy plot, model evaluation results, and Decision Tree visualization.

Running all cells again may produce different results if the dataset changes or the model configuration is modified.

## 📁 Project Structure

A suggested repository structure is:

```text
milk-quality-prediction/
│
├── main.ipynb       # Data analysis, preprocessing, training, and evaluation
├── data.csv         # Milk quality dataset (required by the notebook)
├── README.md        # Project documentation
└── .gitignore       # Files and folders excluded from Git
```

## 🚀 Future Improvements

Potential extensions to the project include:

* **Improved preprocessing:** Fit normalization or scaling only on the training set using a Pipeline.
* **Stratified splitting:** Use `stratify=y` to preserve class proportions in train and test subsets.
* **Expanded evaluation:** Report precision, recall, F1-score, a confusion matrix, and per-class performance.
* **Model comparison:** Evaluate KNN and Decision Tree under the same cross-validation procedure and compare them with additional classifiers.
* **Hyperparameter tuning:** Tune KNN parameters and Decision Tree settings systematically.
* **Reproducibility:** Document the dataset source, Python version, and dependency versions.
* **Prediction interface:** Add a function or simple application that accepts milk characteristics and returns a predicted quality class.

## 👨‍💻 Author

**Sahand Sabet**

* GitHub: [@sahandsbt](https://github.com/sahandsbt)

---

If you find this project useful, consider giving the repository a ⭐ on GitHub!
