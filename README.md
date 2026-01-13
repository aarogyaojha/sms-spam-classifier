# SMS Spam Classifier

## 1. Project Overview
This project is a comprehensive Machine Learning solution designed to detect and filter SMS spam messages. Leveraging Natural Language Processing (NLP) techniques and the **Multinomial Naive Bayes** algorithm, the system classifies messages as either 'Ham' (Legitimate) or 'Spam' with high precision. The project includes a complete data pipeline from exploratory data analysis (EDA) to model deployment via a user-friendly **Streamlit** web application.

## 2. Problem Statement & Motivation
Spam messages are not merely a nuisance; they pose significant security risks, including phishing and financial fraud. The exponential growth of mobile communication has made automated filtering systems a necessity. This project aims to build a robust classifier that minimizes False Positives (legitimate messages marked as spam) while effectively identifying unsolicited content, a critical requirement for any production-grade messaging system.

## 3. Dataset Description
The model is trained on the **SMS Spam Collection Dataset**, a set of 5,572 SMS messages tagged as spam or ham.
- **Source**: UCI Machine Learning Repository.
- **Composition**:
    - **Ham**: ~87%
    - **Spam**: ~13%
- **Preprocessing**: The dataset underwent rigorous cleaning, including duplicate removal and label encoding, ensuring a high-quality input for training.

## 4. Methodological Approach (ML Pipeline)
The project follows a structured data science lifecycle:

1.  **Data Ingestion & Cleaning**: Handling null values and duplicates.
2.  **Exploratory Data Analysis (EDA)**: Visualizing data distributions and analyzing structural differences (e.g., character counts) between spam and ham.
3.  **Text Preprocessing**:
    - **Lowercasing**: To ensure uniformity.
    - **Tokenization**: Breaking text into individual words using NLTK.
    - **Stemming**: Reducing words to their root form (PorterStemmer) to handle variations.
    - **Stopword Removal**: Eliminating non-informative common words.
4.  **Feature Extraction**:
    - **TF-IDF (Term Frequency-Inverse Document Frequency)**: Converting text data into numerical vectors, highlighting unique terms that characterize spam messages.
5.  **Model Training**:
    - Multiple algorithms were benchmarked, including **Logistic Regression, SVM, Random Forest, and Gradient Boosting**.
    - **Naive Bayes (MultinomialNB)** was selected as the final model due to its superior performance on text data.

## 5. Model Evaluation & Results
Given the imbalance in the dataset and the high cost of misclassifying legitimate emails, **Precision** was the primary metric for model selection.

- **Final Model**: Multinomial Naive Bayes
- **Precision Score**: 1.0 (100% Precision on test set)
- **Accuracy Score**: ~97%
- **Confusion Matrix Analysis**: The model demonstrated exceptional ability to distinguish spam without flagging legitimate messages.

## 6. Deployment (Streamlit App)
The best-performing model is deployed using **Streamlit**, allowing users to classify messages in real-time.
- **`app.py`**: Handles the UI, loads the pre-trained vectorizer and model (`vectorizer.pkl`, `model.pkl`), and executes the prediction pipeline.

## 7. How to Run Locally

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Streamlit

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/aarogyaojha/sms-spam-classifier.git
   cd sms-spam-classifier
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Execution
1. **Model Training (Optional)**:
   Run the Jupyter Notebook `sms-spam-detection.ipynb` to retrain the model and generate new pickle files.
   
2. **Run the Application**:
   Execute the following command in your terminal:
   ```bash
   streamlit run app.py
   ```

## 8. Directory Structure
```
sms-spam-classifier/
│
├── sms-spam-detection.ipynb  # Main analysis and training notebook
├── app.py                    # Streamlit deployment script
├── spam.csv                  # Dataset file
├── model.pkl                 # Serialized Naive Bayes model
├── vectorizer.pkl            # Serialized TF-IDF vectorizer
└── README.md                 # Project documentation
```
