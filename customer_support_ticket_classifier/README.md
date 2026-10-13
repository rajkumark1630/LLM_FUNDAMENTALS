Customer Support Ticket Classifier
1. Project Overview

This project focuses on classifying customer support tickets by category and priority using Natural Language Processing (NLP) and machine learning.

This is Part 1 of the project, which covers dataset creation, preprocessing, class balance analysis, and dataset splitting.

2. Dataset Source

A custom synthetic dataset was generated using Python for educational purposes. It does not contain real customer information and was not downloaded from a public dataset.

The dataset contains 594 customer support tickets.

3. Dataset Description

The dataset contains the following columns:

Column	Description
text	Customer support ticket description
category	Category assigned to the ticket
priority	Priority assigned to the ticket
Ticket Categories
Technical
Billing
Account
Delivery
Refund
General Inquiry
Priority Labels
High
Medium
Low
4. Data Preprocessing

The following preprocessing steps were performed:

Converted ticket text to lowercase.
Removed leading and trailing whitespace.
Normalized repeated whitespace.
Removed records with missing or empty values.
Removed duplicate ticket records.
Examined class counts for category and priority.
5. Dataset Summary
Total cleaned records: 594
Number of categories: 6
Number of priority classes: 3
Category Distribution
Category	Count
Technical	99
Billing	99
Account	99
Delivery	99
Refund	99
General Inquiry	99
Priority Distribution
Priority	Count
High	198
Medium	198
Low	198
6. Train, Validation, and Test Split

The dataset was split using stratified sampling with a fixed random state for reproducibility.

Dataset	Percentage	Records
Training	70%	415
Validation	15%	89
Test	15%	90
Total	100%	594

The combined category-priority label was used for stratification to preserve the distribution of both labels as closely as possible.

7. Project Structure
customer-support-ticket-classifier/
├── data/
│   ├── tickets.csv
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
├── src/
│   └── prepare_dataset.py
├── README.md
└── requirements.txt

8. How to Run

Install the dependencies:

python -m pip install -r requirements.txt


Generate the dataset and splits:

python src/prepare_dataset.py

9. Output Files
tickets.csv: Complete cleaned dataset.
train.csv: Training data.
validation.csv: Validation data.
test.csv: Test data.
10. Next Steps

In Part 2, TF-IDF vectorization and Logistic Regression will be used to create baseline classifiers for predicting ticket category and priority.

11. Limitations

The dataset is synthetic and uses predefined text templates. Therefore, model performance on this dataset may not reflect performance on real-world customer support tickets.









Task 2: TF-IDF + Logistic Regression Baseline
Approach

Two separate text classification models were developed:

Category classification
Priority classification

Both models use a TF-IDF vectorizer and Logistic Regression pipeline. Hyperparameters were tuned using 5-fold stratified cross-validation with macro F1 as the scoring metric.

Evaluation
Target	Features	Validation Accuracy	Macro F1
Category	Word TF-IDF	To be filled	To be filled
Priority	Word TF-IDF	To be filled	To be filled
Category	Word + Character TF-IDF	To be filled	To be filled
Priority	Word + Character TF-IDF	To be filled	To be filled
Hyperparameter Tuning
Word n-gram range: (1, 1) and (1, 2)
Logistic Regression C: 0.1, 1.0, and 10.0
Cross-validation: 5 folds
Scoring metric: Macro F1
Model Artifacts

Trained pipelines are saved in the models/ directory using joblib. Evaluation metrics and confusion matrices are saved in reports/.

Error Analysis

The most confusing misclassified validation examples are recorded in reports/top_confusing_examples.csv when validation errors are present.

Limitations

The dataset is synthetic and uses predefined text templates. The evaluation results may not generalize to real-world customer support tickets.


Model Selection and Justification
DistilBERT achieved 98.89% test accuracy and approximately 0.989 macro F1 for both category and priority classification on our synthetic dataset.
Its validation accuracy reached 100%, but this result should be interpreted cautiously because the dataset contains repeated ticket templates.
DistilBERT provides contextual text representations but requires more memory and computational resources than a TF-IDF and Logistic Regression baseline.
The TF-IDF baseline may be preferable for lightweight deployment if its predictive performance is sufficiently close to DistilBERT.
The final deployment choice will be based on validation and test metrics, inference latency, training cost, and resource requirements.




Model Comparison and Selection
Results

The TF-IDF + Logistic Regression baseline achieved 100% accuracy and macro F1 for both category and priority classification in the recorded evaluation results. Word-level unigrams and bigrams were selected through cross-validation. Adding character n-gram features did not improve the reported metrics.

DistilBERT achieved 98.89% test accuracy and approximately 0.989 macro F1 for both classification tasks. Category training took approximately 142.41 seconds, while priority training took approximately 138.15 seconds. Measured inference latency was approximately 17.29 ms and 15.88 ms per ticket, respectively.

Five-line Model Selection Justification
TF-IDF + Logistic Regression achieved the strongest recorded predictive metrics on our synthetic dataset.
DistilBERT achieved 98.89% test accuracy and approximately 0.989 macro F1 for both tasks.
The TF-IDF baseline is likely the more cost-effective deployment option because it is lightweight and requires fewer computational resources.
DistilBERT offers contextual language representations but has higher memory and computational requirements.
We select TF-IDF + Logistic Regression for initial deployment, subject to verifying test-set metrics and evaluating performance on diverse real-world tickets.
Limitations

The dataset contains synthetic ticket templates and repeated linguistic patterns. Consequently, the high scores may overestimate performance on unseen real-world customer messages. Further evaluation on independent, real support tickets is recommended.