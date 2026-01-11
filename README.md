# SMS Spam Classifier

A comprehensive machine learning project for classifying SMS messages as spam or legitimate (ham). This project includes data preprocessing, exploratory data analysis, feature engineering, and advanced text visualization techniques to prepare for building robust spam detection models.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Features](#features)
- [Key Analysis](#key-analysis)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
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

The notebook provides a solid foundation for building and deploying machine learning models for spam classification.

## 📊 Dataset

**Source**: SMS Spam Collection Dataset

**Statistics**:
- Total Messages: 5,572 (before cleaning)
- Unique Messages: 5,169 (after removing duplicates)
- Ham Messages: 4,516 (87.4%)
- Spam Messages: 653 (12.6%)

**Features**:
- Original text messages
- Binary classification labels (ham/spam)
- Engineered features (character count, word count, sentence count)

## 📁 Project Structure

```
sms-spam-classifier/
│
├── sms-spam-detection.ipynb    # Main Jupyter notebook with complete analysis
├── spam.csv                     # SMS Spam Collection dataset
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

## 🔬 Methodology

### 1. Data Loading
- Load SMS dataset with proper encoding (ISO-8859-1)
- Initial data inspection and structure analysis

### 2. Data Cleaning
- Remove unnecessary columns with null values
- Rename columns for clarity (v1 -> target, v2 -> text)
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
- transformed `text` column into `transformed_text` for modeling

### 6. Visual Insights
- **Spam Word Cloud**: Highlighted urgent terms like "FREE", "CALL", "TEXT", "CLAIM"
- **Ham Word Cloud**: Showed conversational terms like "go", "got", "come", "ok"
- **Top 30 Words Analysis**: Confirmed that spam messages rely heavily on call-to-action verbs

## � Installation

### Prerequisites
Ensure you have Python 3.7+ installed on your system.

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd sms-spam-classifier
```

### Step 2: Install Required Packages
```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud jupyter
```

### Step 3: Download NLTK Data
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

## 💻 Usage

### Running the Notebook

1. **Start Jupyter Notebook**:
```bash
jupyter notebook sms-spam-detection.ipynb
```

2. **Execute Cells Sequentially**:
   - Run each cell in order to reproduce the analysis.
   - All visualizations, including Word Clouds, will be generated inline.

## 🔮 Future Enhancements

### Planned Features
- [ ] TF-IDF Vectorization implementation
- [ ] Naive Bayes, SVM, and Random Forest model training
- [ ] Model evaluation (Accuracy, Precision, Recall, F1-Score)
- [ ] Deployment using Streamlit or Flask

## 📦 Dependencies

### Core Libraries
- **pandas** (>=1.3.0): Data manipulation
- **numpy** (>=1.21.0): Numerical computing
- **scikit-learn** (>=0.24.0): Preprocessing and modeling

### NLP & Visualization
- **nltk** (>=3.6.0): Natural language processing
- **wordcloud** (>=1.8.0): Text visualization
- **matplotlib** (>=3.4.0): Plotting
- **seaborn** (>=0.11.0): Statistical styling

### Development
- **jupyter** (>=1.0.0): Notebook environment

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License
This project is created for educational purposes.

## 📧 Contact
For questions or feedback, please open an issue in the repository.
