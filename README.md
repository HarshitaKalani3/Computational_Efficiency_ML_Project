# Computational_Efficiency_in_Hybrid_Model : Hybrid Decision Tree & Neural Network
## Description
Taken this project idea/ problem statement from the Deep Neural Network and Tabular Data: A Survey submitted to IEEE in June 2022 from the part of Open Research Questions; Compuatational efficiency in Hybrid Model. In this project, there is an implementation of the decision tree from scratch with GPU acceleration using CuPy, which is evaluated using train-test split and k-fold cross-validation, then further on it improves the performance using a hybrid model that is the combination of two things, so involves two major stages:
### Decision Tree Learning-
1. Learns from rule-based splits.
2. Then gives class prediction for each sample data of the dataset.
### A Neural Network(Pytorch, that learns from both original features and decision tree predictions).
1. It receives original features(numerical and categorical features) and decision tree predictions.
2. Then produces the final classidfication output.
The dataset used here is the Adult Income Dataset which predicts whether a person earns >50K or <=50K annually. This project focuses on evaluating a hybrid machine learning framework for binary classification using the Adult Income Dataset. The primary goal is to explore how classical machine learning models and deep learning models can be combined to achieve better predictive performance than using either approach alone.

## Dataset
The Adult Income Dataset is the combination of-
### Numerical features: age, education level, hours per week, etc.
### Categorical features: race, gender, marital status, etc.
### Target variable: "income" renamed to "label"
### Classes: 
    <=50K -> 0
    >50K -> 1
Converted labels to numeric form, removed missing values and one-hot encoded categorical features.

## Features
1. Decision tree making from scratch
2. GPU acceleration from CuPy
3. Test-Train split
4. 10(k)-fold cross validation
5. Hybrid Model
6. Performance measuring: Accuracy and Confusion Metrics.

## Objectives
1. Implementing a Decision Tree from scratch
2. Evaluating the performance of the model
3. Designing a hybrid model
4. Exploring difference between CPU and GPU based computation.

## Technologies Used
Numpy, Cupy, Pandas, Pytorch, Scikit-Learn, Google Colab

## Installation
   !pip install cupy-cuda12x --upgrade
   !pip install torch pandas numpy scikit-learn
### Platform and Python version: 
    Google Colab(here) , Python Version: 3.8+
### Hardware; 
    CPU or GPU acceleration

## Decision Tree Implementation
Built from scratch
Uses entropy and information gain
Uses max_depth, min_samples_split and randomly features are selected along the tree
Main Components are : node class, recursive tree growth by its own and level order traversal.
Example Output:
   Depth 0: Split on 'education-num' at threshold 9
   Depth 1: Split on 'age' at threshold 37
   ...
### Model Evaluation
1. Train-Test splitting (70/30):
Accuracy is computed and confusion matrix is created.
Example- Accuracy: 0.84
Confusion Matrix:
   [tn, fp]
   [fn, tp]
2. k-fold Cross-Validation:
Each fold trains a new decision tree and evaluates on new data
Example Output:
   Fold 1 acc: 0.83
   Fold 2 acc: 0.85
   ...
   Avg acc on 10 folds: 0.84

## Hybrid Model: Decision Tree and Neural Network
We are giving original features and decision tree predictions as input to it then,
Input Layer → 32 → 16 → 1
Activation: ReLU + Sigmoid
Loss: Binary Cross Entropy
Optimizer: Adam
### Hybrid k-fold cross-validation
In this, we evaluate accuracy.
Fold 1 Hybrid Accuracy: 0.87
Fold 2 Hybrid Accuracy: 0.88
...
Average Hybrid Accuracy: 0.88
Decision Tree Accuracy: 0.84
Hence, we can see that the hybrid model performs better than the decision tree model.

## How To Run
1. Upload adult.csv to Google Colab.
2. Install all the dependencies and required libraries.
3. Run all the cells sequentially.
4. Check the outputs of each cells, check tree splits feature, accuracy scores, confusion matrix, k-fold cross-validation and hybrid model improvement.

## Conclusion
1. Developed a Decision Tree from scratch.
2. Evaluated model using train-test split and k-fold cross-validation.
3. Built hybrid model.
4. Showed the improvement by the use of machine learning hybrid model on the real-world tabular dataset by improving the fairness.

## Author
### Harshita Kalani - https://github.com/HarshitaKalani3

## Acknowledgement
### Under the guidance of: Prof. Ritambhra Korpal Ma'am
