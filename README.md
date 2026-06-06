# Titanic Survival Classifier

Binary classification project predicting Titanic passenger survival using the Kaggle Titanic dataset. Built as part of an applied ML review sprint alongside Purdue MSDS coursework.

## Project Overview
Using passenger data (class, sex, age, fare, cabin, etc.), the goal is to predict whether a passenger survived the Titanic disaster. This is a binary classification problem (Survived: 0 or 1).

## Approach

### Exploratory Data Analysis
- Analyzed survival rates across all features using bar charts and boxplots
- Identified Sex, Pclass, and Cabin as strongest predictors
- Found multicollinearity between Pclass, Fare, and Cabin

### Feature Engineering
- Imputed missing Age values using mean age grouped by Pclass
- Converted Cabin to binary (has cabin recorded or not)
- Encoded Sex as binary (0=male, 1=female)
- Created Dependents feature (SibSp + Parch)

### Feature Selection
- Used backward elimination via logistic regression p-values
- Final predictors: Pclass, Sex, Age, Cabin, Dependents

### Models Trained
| Model | Error Rate |
|-------|------------|
| Logistic Regression | 18.44% |
| Linear Discriminant Analysis | 18.44% |
| Quadratic Discriminant Analysis | 21.79% |
| Naive Bayes | 22.91% |
| KNN (k=13) | 22.30% |

## Results
- Best model: Logistic Regression
- Kaggle public leaderboard score: 0.761

## Tech Stack
- Python, pandas, numpy, matplotlib, seaborn
- statsmodels, scikit-learn

## Future Work
- SVM
- Ensemble methods (Random Forest, Gradient Boosting)
- Deep learning
