# 🏠 California House Price Prediction

A beginner Machine Learning project focused on predicting median house values in California using Python and Scikit-learn.

This project explores the complete foundational regression workflow — from Exploratory Data Analysis (EDA) and data preprocessing to training multiple regression algorithms, evaluating their performance, and understanding model behaviour through visualizations.

The primary objective was to understand how different regression algorithms learn patterns from data, how model complexity affects performance, and why ensemble learning techniques such as Random Forest can outperform individual Decision Trees.

---

## 📌 Project Overview

| Detail | Description |
|---|---|
| Project | California House Price Prediction |
| Domain | Machine Learning |
| Problem Type | Supervised Learning |
| Learning Task | Regression |
| Dataset | California Housing Dataset |
| Language | Python |
| ML Library | Scikit-learn |
| Best Model | Random Forest Regressor |
| Best Test R² | 0.8075 |

### Project Objectives

- Understand the fundamentals of supervised machine learning and regression.
- Perform Exploratory Data Analysis to understand the dataset.
- Investigate relationships between input features and house values.
- Implement and compare multiple regression algorithms.
- Understand overfitting and model generalization.
- Evaluate models using different regression metrics.
- Visualize prediction errors and model performance.

---

## 📊 Dataset

This project uses the **California Housing Dataset**, available through `sklearn.datasets.fetch_california_housing()`.

The dataset contains information about California housing districts, derived from the 1990 U.S. Census.

### Dataset Information

| Property | Value |
|---|---|
| Total Samples | 20,640 |
| Input Features | 8 |
| Target Variable | MedHouseVal |
| Missing Values | None |
| Problem Type | Regression |

### Input Features

The dataset contains eight numerical features:

| Feature | Description |
|---|---|
| MedInc | Median income in a block group |
| HouseAge | Median house age in a block group |
| AveRooms | Average number of rooms per household |
| AveBedrms | Average number of bedrooms per household |
| Population | Block group population |
| AveOccup | Average number of household members |
| Latitude | Geographic latitude |
| Longitude | Geographic longitude |

### Target Variable

`MedHouseVal` represents the median house value for a California district.

**Important:** Target values are expressed in units of $100,000.

For example:
- 1.0 = $100,000
- 2.0 = $200,000
- 5.0 = $500,000

The dataset has an important limitation: house values are capped at approximately 5.00001 ($500,001). This affects predictions for higher-valued properties.

Dataset documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html

---

## 🛠️ Technologies and Libraries

- **Python** – Primary programming language.
- **NumPy** – Numerical computations and array operations.
- **Pandas** – Data manipulation and analysis.
- **Matplotlib** – Data visualization and plotting.
- **Seaborn** – Exploratory data visualization.
- **Scikit-learn** – Machine Learning algorithms, preprocessing and evaluation.

---

## 📁 Project Structure

```text
house-price-prediction/
│
├── src/
│   ├── eda.py
│   ├── prepare_data.py
│   ├── train_model.py
│   ├── train_decision_tree.py
│   └── train_random_forest.py
│
├── .gitignore
├── README.md
└── .venv/
```

### File Descriptions

| File | Purpose |
|---|---|
| `eda.py` | Performs Exploratory Data Analysis and visualizes the dataset |
| `prepare_data.py` | Loads and prepares the dataset, including the train-test split |
| `train_model.py` | Trains and evaluates the Linear Regression model |
| `train_decision_tree.py` | Trains and experiments with Decision Tree Regression |
| `train_random_forest.py` | Trains and evaluates Random Forest Regression |

---

## 🔍 Project Workflow

### 1. Exploratory Data Analysis (EDA)

The first stage involved exploring the dataset to understand its structure and identify relationships between different features.

Activities included:

- Loading the California Housing dataset.
- Inspecting the dataset using Pandas.
- Understanding feature distributions.
- Exploring descriptive statistics.
- Examining correlations between numerical features.
- Visualizing relationships using graphs.

Correlation analysis helped investigate how different housing characteristics relate to median house values.

### 2. Data Preparation

The dataset was prepared for machine learning by separating the input features and target variable.

- `X` contains the eight input features.
- `y` contains the target variable, `MedHouseVal`.

The dataset was divided into training and testing sets.

| Dataset | Samples |
|---|---:|
| Training Set | 16,512 |
| Testing Set | 4,128 |
| Total | 20,640 |

This corresponds to an 80:20 train-test split.

The same prepared datasets were reused across the different regression experiments to make model comparisons consistent.

### 3. Linear Regression

Linear Regression was implemented as the baseline model.

It attempts to learn a linear relationship between the input features and the target variable.

The model was trained and evaluated using R².

**Purpose:** Establish a baseline against which more complex regression algorithms could be compared.

### 4. Decision Tree Regression

A Decision Tree Regressor was implemented to investigate how a nonlinear model performs on the dataset.

Experiments were conducted using different values of `max_depth`.

| Maximum Depth | Training R² | Testing R² |
|---:|---:|---:|
| 5 | 0.6377 | 0.3697 |
| 10 | 0.8348 | 0.6127 |
| 15 | 0.9607 | 0.6360 |
| 20 | 0.9957 | 0.6318 |

**Observations:**

- Increasing tree depth initially improved testing performance.
- At depth 15, the model achieved its best test R² of approximately 0.6360.
- At depth 20, training performance increased significantly while testing performance decreased slightly.
- This demonstrated the effect of overfitting.

The experiment highlighted the importance of balancing model complexity and generalization.

### 5. Random Forest Regression

Random Forest Regression was implemented using `RandomForestRegressor` from Scikit-learn.

Random Forest is an ensemble learning algorithm that combines multiple Decision Trees and averages their predictions for regression tasks.

The model was initially trained using 100 estimators.

Further experiments were conducted by changing `n_estimators`.

| Number of Trees | Training R² | Testing R² |
|---:|---:|---:|
| 100 | 0.9735 | 0.8044 |
| 200 | 0.9742 | 0.8058 |
| 500 | 0.9746 | 0.8075 |

**Observations:**

- Random Forest significantly outperformed the individual Decision Tree.
- Increasing the number of estimators produced smaller improvements in testing performance.
- The 500-tree configuration achieved the best test R² in these experiments.
- The difference between training and testing performance indicated that some overfitting remained.

---

## 📈 Model Evaluation

The models were evaluated using multiple regression metrics.

### 1. R² Score (Coefficient of Determination)

R² measures how much of the variation in the target variable is explained by the model.

A higher R² generally indicates better explanatory performance.

The final Random Forest achieved:

- Training R²: **0.9746**
- Testing R²: **0.8075**

The testing score indicates that the model explains approximately 80.75% of the variation in the unseen test targets.

R² does not represent the percentage of individually correct predictions.

### 2. Mean Absolute Error (MAE)

MAE calculates the average absolute difference between actual and predicted values.

Final Random Forest result:

**MAE = 0.3259**

Since the target is measured in units of $100,000, this corresponds to an average absolute error of approximately $32,591.

### 3. Mean Squared Error (MSE)

MSE calculates the average squared difference between actual and predicted values.

Final Random Forest result:

**MSE = 0.2523**

Squaring the errors gives larger prediction mistakes more influence on the metric.

### 4. Root Mean Squared Error (RMSE)

RMSE is the square root of MSE.

Final Random Forest result:

**RMSE = 0.5023**

This corresponds to approximately $50,228 in the original target-value units.

---

## 📉 Model Visualizations

Two visualizations were used to understand the behaviour of the final Random Forest model.

### 1. Residual Plot

Residuals represent the difference between actual and predicted values.

```python
residuals = y_test - y_pred
```

The residual plot was used to investigate:

- Whether prediction errors were distributed around zero.
- Whether the model systematically overestimated or underestimated values.
- Whether prediction error variability changed with predicted house values.
- Whether unusual prediction errors were present.

The plot showed a noticeable variation in residual spread and a diagonal boundary associated with the dataset's capped target values.

### 2. Actual vs Predicted Plot

This visualization compares actual house values against the values predicted by the model.

The ideal prediction line represents:

```text
Predicted Value = Actual Value
```

Points closer to the diagonal indicate more accurate predictions.

The visualization showed a strong overall relationship between actual and predicted values, with larger deviations for some higher-valued properties.

---

## 🏆 Final Model Comparison

The following table summarizes the experiments.

| Algorithm | Configuration | Test R² |
|---|---|---:|
| Linear Regression | Default | ~0.56 |
| Decision Tree | max_depth=5 | 0.3697 |
| Decision Tree | max_depth=10 | 0.6127 |
| Decision Tree | max_depth=15 | 0.6360 |
| Decision Tree | max_depth=20 | 0.6318 |
| Random Forest | 100 estimators | 0.8044 |
| Random Forest | 200 estimators | 0.8058 |
| Random Forest | 500 estimators | **0.8075** |

### Best Performing Model

**Random Forest Regressor (500 estimators)**

| Evaluation Metric | Result |
|---|---:|
| Training R² | 0.9746 |
| Testing R² | 0.8075 |
| MAE | 0.3259 |
| MSE | 0.2523 |
| RMSE | 0.5023 |

Based on the experiments conducted, Random Forest achieved the best test performance.

---

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.x
- pip
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/pal-debarpan/house-price-prediction.git
```

```bash
cd house-price-prediction
```

### 2. Create a Virtual Environment

On macOS:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 4. Run the Project

Run the exploratory data analysis:

```bash
python src/eda.py
```

Prepare the dataset:

```bash
python src/prepare_data.py
```

Train Linear Regression:

```bash
python src/train_model.py
```

Train Decision Tree:

```bash
python src/train_decision_tree.py
```

Train Random Forest:

```bash
python src/train_random_forest.py
```

The California Housing dataset can be downloaded automatically through Scikit-learn when it is first fetched.

Run commands from the project root directory.

---

## ⚠️ Limitations

Although the Random Forest achieved a test R² of approximately 0.8075, the project has several limitations:

1. **Overfitting:** The training R² is higher than the testing R², indicating a generalization gap.
2. **Dataset limitation:** The target values are capped at approximately $500,001, limiting the model's ability to learn about higher-valued properties.
3. **Geographical limitation:** The dataset represents California housing districts and should not be treated as a general house-price prediction model for other locations.
4. **Historical data:** The dataset is derived from the 1990 U.S. Census and does not represent current housing market prices.
5. **Model optimization:** The experiments used manually selected hyperparameters. Systematic hyperparameter optimization and cross-validation were not implemented.

The model is intended for learning and experimentation, not real-world property valuation.

---

## 🎯 Key Learnings

Through this project, I explored and practised:

- Fundamentals of supervised machine learning.
- Regression algorithms and their applications.
- Exploratory Data Analysis using Pandas and visualization libraries.
- Data preparation and train-test splitting.
- Linear Regression, Decision Trees and Random Forests.
- Hyperparameters and model complexity.
- Overfitting and generalization.
- Ensemble learning and Bootstrap Aggregating.
- Regression evaluation metrics: R², MAE, MSE and RMSE.
- Residual analysis and Actual vs Predicted visualization.

Most importantly, this project helped me understand that a more complex model does not automatically produce better predictions, and that evaluating performance on unseen data is essential.

---

## 🚀 Future Improvements

Potential extensions to this project include:

- Hyperparameter optimization using GridSearchCV.
- Cross-validation for more reliable model evaluation.
- Feature importance analysis.
- Experimenting with additional regression algorithms.
- Saving trained models using Joblib.
- Deploying the model using FastAPI.
- Building a frontend for interactive house-price predictions.

These are possible future extensions and are not part of the current implementation.

---

## 👨‍💻 Author

**Debarpan Pal**

B.Tech Computer Science and Engineering  
Vellore Institute of Technology, Vellore

- GitHub: [pal-debarpan](https://github.com/pal-debarpan)
- LinkedIn: [Debarpan Pal](https://www.linkedin.com/in/debarpan-pal-488998414)

---

*This project was developed as a hands-on learning experience to understand the fundamentals of Machine Learning and regression modelling.*