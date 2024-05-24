# Heart Disease Prediction Project

This repository contains a comprehensive project for predicting heart disease using various machine learning models. The dataset used is a combination of the Heart Disease datasets from Statlog, Cleveland, and Hungary.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Modeling](#modeling)
  - [Logistic Regression](#logistic-regression)
  - [Naive Bayes](#naive-bayes)
  - [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
  - [Improved Logistic Regression](#improved-logistic-regression)
- [Evaluation](#evaluation)
  - [Confusion Matrix and Classification Report](#confusion-matrix-and-classification-report)
  - [K-Fold Cross Validation](#k-fold-cross-validation)
- [Usage](#usage)
- [Conclusion](#conclusion)
- [License](#license)

## Introduction
This project aims to predict the presence of heart disease in patients using machine learning models. It covers data preprocessing, exploratory data analysis, model training, and evaluation.

## Dataset
The dataset used in this project is a combination of the Heart Disease datasets from Statlog, Cleveland, and Hungary. It includes features such as:
- `age`: Age of the patient
- `sex`: Sex of the patient (1 = male, 0 = female)
- `chest pain type`: Type of chest pain experienced
- `resting bp s`: Resting blood pressure
- `cholesterol`: Serum cholesterol in mg/dl
- `fasting blood sugar`: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- `resting ecg`: Resting electrocardiographic results
- `max heart rate`: Maximum heart rate achieved
- `exercise angina`: Exercise-induced angina (1 = yes, 0 = no)
- `oldpeak`: ST depression induced by exercise relative to rest
- `ST slope`: The slope of the peak exercise ST segment
- `target`: Diagnosis of heart disease (1 = presence, 0 = absence)

## Dependencies
To run this project, you need the following libraries:
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn

You can install these dependencies using pip:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

## Data Preprocessing
The dataset is loaded and preprocessed as follows:
1. **Loading the Data**: The dataset is loaded into a pandas DataFrame.
   ```python
   csv_file = pd.read_csv('heart_statlog_cleveland_hungary_final.csv')
   initial_dataframe = pd.DataFrame(csv_file)
   ```
2. **Separating Numerical and Categorical Features**: The numerical and categorical features are separated for easier analysis and visualization.
   ```python
   numerical_df = initial_dataframe[['age','resting bp s','cholesterol','max heart rate','oldpeak']]
   categorial_df = initial_dataframe[['sex','chest pain type','fasting blood sugar','resting ecg','exercise angina','ST slope']]
   ```
3. **Defining the Plotting Function**: A function `plotting` is defined for visualizing categorical and numerical features, correlation matrix, and target evaluation.
   ```python
   def plotting(data, subdata, data_type): 
       if data_type == 'categorial':
           fig, axes = plt.subplots(2, 3, figsize=(20, 20))
           for i, axes in enumerate(axes.flatten()):
               sns.countplot(x=subdata.columns[i], data=data, ax=axes)
       elif data_type == 'numerical':
           fig, axes = plt.subplots(2, 3, figsize=(20, 20))
           for i, axes in enumerate(axes.flatten()):
               if i <= 4:
                   sns.boxplot(x=subdata.columns[i], data=data, ax=axes)
       elif data_type == 'correlation':
           plt.figure(figsize=(10, 10))
           plt.title('Correlation Matrix')
           plt.xticks(fontsize=10)
           plt.yticks(fontsize=10)
           sns.heatmap(data=data.corr(), cmap='Reds', annot=True, linewidths=2)
       elif data_type == 'target-evaluation':
           for i in numerical_df.columns: 
               title = i + ' ' + 'vs' + ' ' + 'target'
               x_label = i
               y_label = 'target'
               plt.figure(figsize=(15, 8))
               plt.scatter(x=numerical_df[i], y=data['target'])
               plt.xlabel(x_label)
               plt.ylabel(y_label)
               plt.title(title)
   ```
4. **Removing Noisy Data**: Rows with zero cholesterol values are removed as they are considered noise.
   ```python
   noise_1 = initial_dataframe[initial_dataframe['cholesterol'] == 0].index
   initial_dataframe.drop(axis=0, index=noise_1, inplace=True)
   numerical_df.drop(axis=0, index=noise_1, inplace=True)
   ```

## Exploratory Data Analysis (EDA)
EDA is performed using the `plotting` function to visualize:
- **Distribution of Categorical Features**: Count plots are created for categorical features to visualize their distribution.
   ```python
   plotting(initial_dataframe, subdata=categorial_df, data_type='categorial')
   ```
- **Distribution and Outliers in Numerical Features**: Box plots are created for numerical features to identify outliers and their distribution.
   ```python
   plotting(data=initial_dataframe, subdata=numerical_df, data_type='numerical')
   ```
- **Correlation Between Features**: A correlation matrix is generated to identify the relationships between features.
   ```python
   plotting(data=initial_dataframe, subdata=None, data_type='correlation')
   ```
- **Relationship Between Numerical Features and Target Variable**: Scatter plots are created to visualize the relationship between numerical features and the target variable.
   ```python
   plotting(data=initial_dataframe, subdata=numerical_df, data_type='target-evaluation')
   ```

## Modeling
### Logistic Regression
A logistic regression model is applied to the dataset using the `model_application` function, which tests various train-test splits and selects the best model based on accuracy.
```python
def model_application(X, Y, model):
    test_size = [0.1, 0.15, 0.2, 0.25]
    initial_accuracy = 0
    for i in test_size:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y.ravel(), test_size=i, random_state=0)
        model.fit(X_train, Y_train)
        Y_prediction = model.predict(X_test)
        final_accuracy = accuracy_score(Y_test, Y_prediction)
        if final_accuracy > initial_accuracy:
            initial_accuracy = final_accuracy
            final_model = model
            best_test_size = i
            final_Y_test = Y_test
            final_Y_prediction = Y_prediction
            final_X_train = X_train
            final_Y_train = Y_train
    return model, final_Y_test, final_Y_prediction, final_X_train, final_Y_train
model, Y_test, Y_prediction, X_train, Y_train = model_application(X=X, Y=Y, model=LogisticRegression())
```

### Naive Bayes
A Gaussian Naive Bayes model is applied similarly using the `model_application` function.
```python
model, Y_test, Y_prediction, X_train, Y_train = model_application(X=X, Y=Y, model=GaussianNB())
```

### K-Nearest Neighbors (KNN)
A KNN model is tested with different values of k using the `knn` function. The best k value is selected based on model accuracy.
```python
def knn(X, Y, k):
    Training_acc_list = []
    test_acc_list = []
    initial_accuracy = 0
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=0)
    k_values = [i for i in range(3, k, 2)]
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, Y_train.ravel())
        Y_prediction = model.predict(X_test)
        final_accuracy = accuracy_score(Y_test, Y_prediction)
        Training_acc_list.append(model.score(X_train, Y_train))
        test_acc_list.append(model.score(X_test, Y_test))
        if final_accuracy > initial_accuracy:
            initial_accuracy = final_accuracy
            final_model = model
            final_k = k
            final_X_train, final_X_test, final_Y_train, final_Y_test, final_Y_prediction = X_train, X_test, Y_train, Y_test, Y_prediction
    plt.plot(k_values, Training_acc_list, label='Accuracy of training set')
    plt.plot(k_values, test_acc_list, label='Acuuracy of test set')
    plt.xlabel('Number of the neighbours')
    plt.ylabel('Accuracy')
    plt.xticks(ticks=k_values)
    plt.grid()
    plt.legend()
    return model, final_k, final_X_train, final_X_test, final_Y_train, final_Y_test, final_Y_prediction
model, k, X_train, X_test, Y_train, Y_test, Y_prediction = knn(X=X, Y=Y, k=50)
```

### Improved Logistic Regression
An improved logistic regression model

 is built using different solvers and regularization parameters. The best model is selected based on accuracy.
```python
def improved_logestic_model(X, Y):
    initial_accuracy_score = 0
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y.ravel(), test_size=0.15, random_state=0)
    solver = ['saga', 'sag', 'newton-cholesky', 'newton-cg', 'liblinear', 'lbfgs']
    c_value = [0.1, 0.05, 0.01, 0.001, 0.0001]
    max_iter = 10000
    for i in range(len(solver)):
        accuracy_list = {}
        for j in range(len(c_value)):
            model = LogisticRegression(solver=solver[i], C=c_value[j], max_iter=max_iter)
            model.fit(X_train, Y_train)
            Y_prediction = model.predict(X_test)
            final_accuracy_score = accuracy_score(Y_test, Y_prediction)
            accuracy_list[(solver[i], c_value[j])] = final_accuracy_score
            if final_accuracy_score > initial_accuracy_score:
                initial_accuracy_score = final_accuracy_score
                final_model_name = solver[i]
                final_c_value = c_value[j]
                final_model = model
                final_Y_test = Y_test
                final_Y_prediction = Y_prediction
        for key, value in accuracy_list.items():
            print(f'for the c = {key[1]} : the accuracy is equal to : {value}')
    return final_model, final_Y_test, final_Y_prediction
model, Y_test, Y_prediction = improved_logestic_model(X=X, Y=Y)
```

## Evaluation
### Confusion Matrix and Classification Report
The `confusion_evaluation` function is used to evaluate the model's performance by displaying the confusion matrix and classification report.
```python
def confusion_evaluation(Y_test, Y_prediction): 
    result_confusion = confusion_matrix(Y_test, Y_prediction)
    result_classification = classification_report(Y_test, Y_prediction)
    print(result_confusion)
    print(result_classification)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in result_confusion.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in result_confusion.flatten()/np.sum(result_confusion)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    heatmap = sns.heatmap(result_confusion, annot=labels, fmt='', cmap='Blues')
    heatmap.set_xlabel("Prediction")
    heatmap.set_ylabel('True')
confusion_evaluation(Y_test=Y_test, Y_prediction=Y_prediction)
```

### K-Fold Cross Validation
The `folds_evaluation` function performs k-fold cross-validation to evaluate the model's performance across different folds.
```python
def folds_evaluation(X, Y, folds_number):
    kf = KFold(n_splits=folds_number, shuffle=False, random_state=None)
    cv = cross_val_score(estimator=model, X=X_train, y=Y_train, cv=kf)
    for i in range(len(cv)):
        print(f'The Accuracy for the fold {i+1} is equal to : {cv[i]}')
    print(f'The average of the accuracy is equal to {np.average(cv)}')
    print(f'The number of the data in each fold is equal to {int(len(X_train)/folds_number)}')
folds_evaluation(X=X_train, Y=Y_train, folds_number=10)
```

## Usage
To use this project, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/heart-disease-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd heart-disease-prediction
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the main script to load data, preprocess, perform EDA, train models, and evaluate them.

## Conclusion
This project demonstrates the use of machine learning models for predicting heart disease. It covers data preprocessing, visualization, model training, and evaluation. Future work can include hyperparameter tuning, feature engineering, and testing additional models.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to contribute to this project by opening issues or submitting pull requests. For any questions, please contact [your email].
 
