# SMS Spam Classifier

A comprehensive machine learning project for classifying SMS messages as spam or legitimate (ham). This project includes data preprocessing, exploratory data analysis, feature engineering, text visualization, model building, and model comparison to create a robust spam detection system.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Features](#features)
- [Key Analysis](#key-analysis)
- [Models Used](#models-used)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

This project implements a complete data science pipeline for SMS spam detection, including:
- **Data Cleaning**: Handling missing values and duplicates
- **Feature Engineering**: Creating statistical features from text data
- **Text Preprocessing**: NLP pipeline (tokenization, stemming, etc.)
- **Visual Analysis**: Word clouds and frequency distributions
- **Exploratory Data Analysis**: Understanding data patterns and distributions
- **Model Building**: Training and comparing multiple machine learning models
- **Model Evaluation**: Comprehensive evaluation using accuracy and precision metrics

The notebook provides a complete workflow from raw data to a deployed-ready machine learning model for spam classification.

## 📊 Dataset

**Source**: SMS Spam Collection Dataset

**Statistics**:
- Total Messages: 5,572 (before cleaning)
- Unique Messages: 5,169 (after removing 403 duplicates)
- Ham Messages: 4,516 (87.4%)
- Spam Messages: 653 (12.6%)

**Features**:
- Original text messages
- Binary classification labels (ham/spam)
- Engineered features (character count, word count, sentence count)
- Transformed text (preprocessed using NLP techniques)

## 📁 Project Structure

```
sms-spam-classifier/
│
├── sms-spam-detection.ipynb    # Main Jupyter notebook with complete analysis
├── spam.csv                     # SMS Spam Collection dataset
├── model.pkl                    # Trained model (Multinomial Naive Bayes)
├── vectorizer.pkl               # TF-IDF vectorizer for text transformation
├── README.md                    # Project documentation (this file)
└── .git/                        # Git repository
```

## ✨ Features

### Data Processing
- ✅ Automatic handling of missing values
- ✅ Duplicate message detection and removal
- ✅ Label encoding (ham → 0, spam → 1)
- ✅ Text encoding handling (ISO-8859-1)

### Feature Engineering
1. **num_characters**: Total character count per message
2. **num_words**: Word count using NLTK tokenization
3. **num_sentences**: Sentence count using NLTK sentence tokenizer

### Text Preprocessing Pipeline
- **Lowercasing**: Converting all text to lowercase
- **Tokenization**: Breaking text into individual words
- **Special Character Removal**: Stripping non-alphanumeric characters
- **Stopword Removal**: Filtering out common words (e.g., 'is', 'the', 'of')
- **Stemming**: Reducing words to their root forms using PorterStemmer

### Visual Analysis
- **Word Clouds**: Visualizing the most prominent words in Spam and Ham messages
- **Frequency Analysis**: Top 30 most recurring words for each class

### Feature Extraction
- **TF-IDF Vectorizer**: Converting text into numerical features (3000 features)
- Maximum of 3000 most important features selected

## 🔬 Methodology

### 1. Data Loading
- Load SMS dataset with proper encoding (ISO-8859-1)
- Initial data inspection and structure analysis

### 2. Data Cleaning
- Remove unnecessary columns with null values
- Rename columns for clarity (v1 → target, v2 → text)
- Handle missing values and remove duplicate messages (403 duplicates removed)

### 3. Feature Engineering
- **Character Count**: Measure message length
- **Word Count**: Analyze vocabulary usage
- **Sentence Count**: Understand message structure

### 4. Exploratory Data Analysis (EDA)
- **Class Distribution**: Analyzed the imbalance between spam (12.6%) and ham (87.4%)
- **Feature Relationships**: Visualized correlation between message length and spam probability (Spam messages tend to be longer)

### 5. Text Preprocessing
- Applied NLTK-based preprocessing pipeline to clean raw text
- Transformed `text` column into `transformed_text` for modeling

### 6. Visual Insights
- **Spam Word Cloud**: Highlighted urgent terms like "FREE", "CALL", "TEXT", "CLAIM"
- **Ham Word Cloud**: Showed conversational terms like "go", "got", "come", "ok"
- **Top 30 Words Analysis**: Confirmed that spam messages rely heavily on call-to-action verbs

### 7. Model Building
- **Feature Extraction**: TF-IDF Vectorization with 3000 features
- **Train-Test Split**: 80% training, 20% testing
- **Model Training**: Trained multiple classification algorithms
- **Model Evaluation**: Compared models using accuracy and precision metrics

## 🎯 Models Used

The following machine learning algorithms were trained and evaluated:

1. **Naive Bayes Variants**
   - Gaussian Naive Bayes (GNB)
   - Multinomial Naive Bayes (MNB) ⭐ **Best Model**
   - Bernoulli Naive Bayes (BNB)

2. **Ensemble Methods**
   - Random Forest (RF)
   - Extra Trees Classifier (ETC)
   - Bagging Classifier (BgC)
   - AdaBoost
   - Gradient Boosting (GBDT)
   - XGBoost (xgb)

3. **Other Classifiers**
   - Support Vector Classifier (SVC)
   - Logistic Regression (LR)
   - Decision Tree (DT)
   - K-Neighbors Classifier (KN)

## 📈 Results

### Best Performing Models (Accuracy):
1. **Random Forest**: 97.58%
2. **SVC**: 97.58%
3. **Extra Trees**: 97.49%
4. **Multinomial Naive Bayes**: 97.10% ⭐ *Selected for deployment*
5. **XGBoost**: 96.71%

### Best Performing Models (Precision):
1. **K-Neighbors**: 100%
2. **Multinomial Naive Bayes**: 100% ⭐ *Selected for deployment*
3. **Random Forest**: 98.29%
4. **Extra Trees**: 97.46%
5. **SVC**: 97.48%

**Selected Model**: Multinomial Naive Bayes was chosen as the final model because:
- Perfect precision (100%) - No false positives
- High accuracy (97.10%)
- Computationally efficient
- Well-suited for text classification tasks
- Fast prediction times

### Model Files
- `model.pkl`: Trained Multinomial Naive Bayes model
- `vectorizer.pkl`: TF-IDF vectorizer for text transformation

## 💻 Installation

### Prerequisites
Ensure you have Python 3.7+ installed on your system.

### Step 1: Clone the Repository
```bash
git clone https://github.com/aarogyaojha/sms-spam-classifier
cd sms-spam-classifier
```

### Step 2: Install Required Packages
```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud jupyter xgboost
```

### Step 3: Download NLTK Data
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

## 📊 Usage

### Running the Notebook

1. **Start Jupyter Notebook**:
```bash
jupyter notebook sms-spam-detection.ipynb
```

2. **Execute Cells Sequentially**:
   - Run each cell in order to reproduce the analysis
   - All visualizations, including Word Clouds, will be generated inline
   - Model training and evaluation results will be displayed

### Using the Trained Model

```python
import pickle

# Load the trained model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Preprocess and predict
def predict_spam(message):
    # Apply the same preprocessing as in training
    # (tokenize, remove stopwords, stem)
    processed_message = preprocess(message)  # Use your preprocessing function
    
    # Transform using TF-IDF
    features = vectorizer.transform([processed_message])
    
    # Predict
    prediction = model.predict(features)
    
    return "Spam" if prediction[0] == 1 else "Ham"

# Example
message = "Free entry to win cash prize! Call now!"
print(predict_spam(message))  # Output: Spam
```
## 📦 Dependencies

### Core Libraries
- **pandas** (>=1.3.0): Data manipulation and analysis
- **numpy** (>=1.21.0): Numerical computing and array operations
- **scikit-learn** (>=0.24.0): Machine learning algorithms and tools

### NLP & Visualization
- **nltk** (>=3.6.0): Natural language processing toolkit
- **wordcloud** (>=1.8.0): Text visualization through word clouds
- **matplotlib** (>=3.4.0): Plotting and data visualization
- **seaborn** (>=0.11.0): Statistical data visualization

### Machine Learning
- **xgboost** (>=1.5.0): Gradient boosting framework

### Development
- **jupyter** (>=1.0.0): Interactive notebook environment

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### How to Contribute
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is created for educational purposes.

## 📧 Contact

For questions or feedback, please open an issue in the repository.

