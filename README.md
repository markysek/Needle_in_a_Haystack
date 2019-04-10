# Needle in a Haystack - Fraud Detection project summary 
(for code and visualizations please see Jupyter NTB)

## Abstract
The ultimate goal of the project is to develop a counter-solution to account takeover attacks in payment services. The primary goal is to build a model that could identify patterns in data and spot fraudulent transactions with the highest hit-rate possible. The secondary goal is to find use of it for businesses based on their risk-aversion level:
- Send suspicious transactions to further human investigation, or
- Completely block flagged transactions

## Source of data
The lack of publicly available data on financial transactions was addressed by group TESTIMON when they introduced a simulator PaySim that creates synthetic transaction data based on a sample of real transactions from a mobile money service implemented in an unspecified African country. The synthetic dataset resembles the normal operation of transactions with injected malicious behavior.

The data is available on Kaggle under the name Synthetic Financial Datasets For Fraud Detection. It was published by TESTIMON which is the Digital Forensics Research Group from NTNU in Gjøvik, Norway. There are various academic papers published by authors of the data simulator that help to understand what the data represent.

## Assumptions and Considerations
### 1) Class-imbalance
Presence of fraudulent transactions tends to be rare and it is the case for given dataset as well. When trying to build the best model, I considered three different scenarios:
- oversampled train set & oversampled test set
- non-oversampled train set & non-oversampled test set
- oversampled train set & non-oversampled test set
Even though the models' performance on oversampled train and test sets was better, I decided to maintain the original proportion of fraudulent to genuine transactions for two reasons:
- the goal is to find a model that could be used in real-world conditions
- in case of unlabeled data, it wouldn't be clear which transactions to oversample

### 2) Categorical variables with high cardinality
Two of the original features are counterparties involved in the transaction, namely, 'nameOrig' being the unique identifier of the client where transaction originated, and the 'nameDest' identifying the recipient of given transaction.

My initial approach was to encode these high-cardinality variables using Weight of Evidence encoding algorithm that is based on correlation of the categorical attributes to be encoded to the target variables. The assigned/encoded numerical value is a function of number of records with the categorical value in question and how they break down between positive and negative class attribute values. Additionally, total number of records with the positive and negative class labels are also taken into account. For class imbalanced data, Weight of Evidence algorithm should work better than let's say Target or Leave-one-out encoder.

I also wanted to use the 'nameOrig' and the 'nameDest' as a common key for identifying transactions where cash-out followed transfer, since the type of fraud present in this dataset is account takeover, but my assumption proved to be incorrect, at least for this dataset, and I couldn't link intuitively connected transactions.

The final decision was not to use 'nameOrig' and 'nameDest' due to the fact that fraudsters would most probably not use the same accounts for withdrawing money from the system repetitively.

### 3) Various models
For each of the oversampled/not-oversampled and encoded/not-encoded scenarios, 5 different classifier models or ensembles of models were used:
- logistic regression
- decision tree
- random forest
- XGBoost using decision tree as a base model
- multi-layer perceptron
- The random forest model and the XGBoost delivered the best results, hence the notebook focuses only on the two beforehand mentioned.

## Final approach and the results
Initially, 5 models were fed with original set of data (base case). Additional 7 features were created with the aim to achieve better model classification results. The key metrics to measure the performance were Precision, Recall and F1 score. Random Forest and XGBoost models delivered the best results. 

Precision = what percentage of the transactions predicted as being fraudulent were fraudulent in reality
Recall = the percentage of real fraud cases the model is able to catch
F1 score = harmonic mean of Precision and Recall -> to find the right balance of the two metrics

| Model                                     | Accuracy  | Precision | Recall  | F1 score | AUPRC  |
| ----------------------------------------- | --------- | --------- | ------- | ---------|------- |
| XGBoost model - base case scenario        |     99.96 |     97.24 |   68.78 |    80.57 |  66.92 |
| XGBoost model - with engineered features  |    100.00 |     99.03 |  100.00 |    99.51 |  99.03 |


| Confusion matrix                     | Predicted value - genuine transaction  | True value - fraudulent transaction |
| ------------------------------------ | -------------------------------------- | ----------------------------------- | 
| True value - genuine transaction     |                               158 859  |                                   2 |   
| True value - fraudulent transaction  |                                     0  |                                 205 |  

The confusion matrix shows that while 158,859 cases were correctly classified as non-fraudulent and 205 correctly classified as fraudulent, 2 cases got misclassified as being fraudulent when they weren't and 0 fraud cases were missed. 

The seriousness of misclassification depends on the use case as well as on the value of the misclassified transactions. In this case, the model is more conservative and identified two cases as being fraudulent when they were not. The implication in real life would be probably blocking or at least suspending such transactions. It is assumed that payment service provider would like to avoid this type of misclassifications not to create unnecessary friction during customer's shopping experience which could also create a reputational risk.

## Conclusion
The best two models for detecting and predicting fraudulent transactions proved to be Random Forest and XGBoost with a Decision Tree as a base model. While reaching close-to-100% levels of F1 score, Precision, Recall and AUPRC, the interpretability of the model is not as straight-forward as for a logistic regression for example. The next steps would be to use ideally SHAP package using the Shapley values that consistently evaluate the feature importance while not depending on the order in which the features were added/evaluated or how deep in the decision tree are they positioned. The interpretability of the model is one of the key aspects of model validation framework whose intention is, among other, to check whether a model being used by financial institution or a provider of behavioral solution services is compliant with anti-discrimination laws.
