# California Housing Price Prediction with KNN & GridSearchCV

A machine learning project implementing a **K-Nearest Neighbors (KNN) Regressor** optimized via **GridSearchCV** to predict median house values in California districts based on the classic Kaggle dataset.

## 🚀 Project Overview

The goal of this project is to build an accurate regression model to estimate housing prices by leveraging geographic, demographic, and housing-related features. To achieve optimal performance, hyperparameter tuning is performed systematically using Grid Search with cross-validation.

## 📊 Dataset Features

The dataset contains features derived from the 1990 U.S. Census, including:
* **Longitude & Latitude:** Geographic coordinates of the block group.
* **Housing Median Age:** Median age of a house within a block.
* **Total Rooms & Total Bedrooms:** Aggregate counts for the block group.
* **Population:** Total number of people residing in the block group.
* **Households:** Total number of households in the block group.
* **Median Income:** Median income for households within a block group (measured in tens of thousands of US Dollars).
* **Median House Value:** Target variable (dependent variable) representing median house value.

## 🛠️ Tech Stack & Libraries

* **Language:** Python 3.x
* **Core Libraries:** 
  * `pandas`, `numpy` (Data manipulation & preprocessing)
  * `scikit-learn` (Machine Learning models, metrics, and hyperparameter tuning)
  * `matplotlib`, `seaborn` (Data visualization & exploratory data analysis)

## ⚙️ Methodology & Pipeline

1. **Exploratory Data Analysis (EDA):** Inspecting data distributions, handling missing values, and visualizing geographic patterns.
2. **Feature Scaling:** Since KNN relies on distance metrics (e.g., Euclidean distance), features are standardized using `StandardScaler`.
3. **Model Training:** Training an initial `KNeighborsRegressor` baseline.
4. **Hyperparameter Tuning (`GridSearchCV`):** Experimenting with different values of `n_neighbors`, distance metrics (`p` parameter for Manhattan vs. Euclidean), and weights (`uniform` vs. `distance`) to find the best-performing combination.
5. **Evaluation:** Assessing the optimized model using metrics such as **Root Mean Squared Error (RMSE)** and **R-squared ($R^2$) score**.

## 💻 Installation & Usage

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
   cd your-repo-name
   ```

2. Install the required dependencies:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```

3. Run the Jupyter Notebook or Python script to execute the pipeline and see the results.



## 👤 Author
  - Farbod Mehrgan
  - Kaggle: [Kaggle Profile](https://www.kaggle.com/farbodmehrgan)
  - GitHub: [GitHub Profile](https://github.com/WeirdBlueFish)
