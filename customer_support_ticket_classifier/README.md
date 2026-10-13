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