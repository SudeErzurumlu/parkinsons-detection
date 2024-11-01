# Parkinson Disease Classification with Random Forest

## Overview

This project aims to classify Parkinson's disease using a dataset containing various features related to patient health. The model employs the Random Forest algorithm, a popular ensemble method known for its accuracy and robustness in classification tasks.

## Dataset

The dataset used for this project is obtained from Kaggle. It contains features extracted from voice recordings of patients with Parkinson's disease. You can access the dataset [here]((https://www.kaggle.com/datasets/shreyadutta1116/parkinsons-disease/)).

### Dataset Features
- `name`: Identifier for each record (not used in training)
- `status`: Target variable (1 indicates Parkinson's disease, 0 indicates no Parkinson's disease)
- Various features representing vocal attributes

## Libraries Used

- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `scikit-learn`: For machine learning algorithms and evaluation metrics.

## Model Implementation

1. **Data Loading**: The dataset is loaded using `pandas`.
2. **Data Preparation**: 
   - Non-numeric columns are removed, and the target variable is separated.
   - The target variable is converted to an integer type.
3. **Data Splitting**: The dataset is split into training and testing sets with a ratio of 80% for training and 20% for testing.
4. **Data Normalization**: The features are standardized using `StandardScaler` to improve model performance.
5. **Random Forest Classifier**: The Random Forest model is trained on the scaled training data, and predictions are made on the testing data.

## Results

The model achieved impressive performance on the test set, with the following metrics:

- **Random Forest Accuracy**: 0.9916

### Confusion Matrix

```
[[211  2]
 [  0 26]]
```

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 1.00      | 0.99   | 1.00     | 213     |
| 1     | 0.93      | 1.00   | 0.96     | 26      |

- **Accuracy**: 0.99
- **Macro Average**: 
  - Precision: 0.96
  - Recall: 1.00
  - F1-Score: 0.98
- **Weighted Average**: 
  - Precision: 0.99
  - Recall: 0.99
  - F1-Score: 0.99

## Conclusion

The Random Forest model successfully classified Parkinson's disease with high accuracy. The results indicate that the model is particularly effective in identifying patients without the disease (Class 0) while still performing well in identifying those with the disease (Class 1).

## Future Work

- Explore other classification algorithms and compare their performances.
- Implement hyperparameter tuning to optimize the Random Forest model further.
- Investigate feature importance to identify which health metrics are most indicative of Parkinson's disease.


## How to Run the Code

1. Clone this repository:
   ```bash
   git clone https://github.com/SudeErzurumlu/parkinsons-detection.git
   
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
