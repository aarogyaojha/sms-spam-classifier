# SMS Spam Classifier

A comprehensive machine learning project for classifying SMS messages as spam or legitimate (ham). This project includes data preprocessing, exploratory data analysis, feature engineering, and prepares the foundation for building robust spam detection models.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Future Enhancements](#future-enhancements)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

This project implements a complete data science pipeline for SMS spam detection, including:
- **Data Cleaning**: Handling missing values and duplicates
- **Feature Engineering**: Creating statistical features from text data
- **Exploratory Data Analysis**: Understanding data patterns and distributions
- **Visualization**: Comprehensive visual analysis of the dataset

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

### Exploratory Data Analysis
- Class distribution analysis
- Character count distribution comparison
- Pairwise feature relationship visualization
- Statistical summaries

## 🚀 Installation

### Prerequisites
Ensure you have Python 3.7+ installed on your system.

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd sms-spam-classifier
```

### Step 2: Install Required Packages
```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn jupyter
```

### Step 3: Download NLTK Data
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

## 💻 Usage

### Running the Notebook

1. **Start Jupyter Notebook**:
```bash
jupyter notebook sms-spam-detection.ipynb
```

2. **Execute Cells Sequentially**:
   - Run each cell in order to reproduce the analysis
   - All visualizations will be generated inline

### Quick Start Example

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('spam.csv', encoding='ISO-8859-1')

# Clean the data
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)

# Encode labels
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

# Remove duplicates
df.drop_duplicates(keep='first', inplace=True)

print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{df['target'].value_counts()}")
```

## 🔬 Methodology

### 1. Data Loading
- Load SMS dataset with proper encoding (ISO-8859-1)
- Initial data inspection and structure analysis

### 2. Data Cleaning
- Remove unnecessary columns with null values
- Rename columns for clarity
- Handle missing values
- Remove duplicate messages

### 3. Feature Engineering
- **Character Count**: Measure message length
- **Word Count**: Analyze vocabulary usage
- **Sentence Count**: Understand message structure

### 4. Exploratory Data Analysis
- Visualize class distribution (pie chart)
- Compare feature distributions between spam and ham
- Analyze feature correlations using pair plots

### 5. Data Preparation (Next Steps)
- Text preprocessing (lowercasing, removing special characters)
- Stopword removal
- Stemming/Lemmatization
- TF-IDF vectorization

## 📈 Key Findings

1. **Class Imbalance**: 
   - The dataset is heavily imbalanced (87.4% ham vs 12.6% spam)
   - Requires special handling during model training

2. **Message Length Patterns**:
   - Spam and ham messages show different character count distributions
   - This feature can be useful for classification

3. **Data Quality**:
   - Clean dataset after preprocessing
   - No missing values in final dataset
   - 403 duplicate messages removed

## 🔮 Future Enhancements

### Planned Features
- [ ] Advanced text preprocessing
- [ ] TF-IDF vectorization
- [ ] Multiple ML model implementations (Naive Bayes, SVM, Random Forest)
- [ ] Model comparison and evaluation
- [ ] Hyperparameter tuning
- [ ] Cross-validation
- [ ] Model deployment pipeline

### Additional Features to Engineer
- [ ] Presence of URLs
- [ ] Presence of phone numbers
- [ ] Special character ratio
- [ ] Capitalization ratio
- [ ] Presence of currency symbols
- [ ] Message urgency indicators

## 📦 Dependencies

### Core Libraries
- **pandas** (>=1.3.0): Data manipulation and analysis
- **numpy** (>=1.21.0): Numerical computing
- **scikit-learn** (>=0.24.0): Machine learning algorithms

### NLP Libraries
- **nltk** (>=3.6.0): Natural language processing

### Visualization
- **matplotlib** (>=3.4.0): Plotting and visualization
- **seaborn** (>=0.11.0): Statistical data visualization

### Development
- **jupyter** (>=1.0.0): Interactive notebook environment

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is created for educational purposes as part of an SMS Spam Classification task.

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

## 📚 Additional Resources

- [SMS Spam Collection Dataset Information](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- [NLTK Documentation](https://www.nltk.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

---

*Last Updated: January 11, 2026*

**Note**: This project is currently in the exploratory data analysis phase. Machine learning model implementation is planned for future updates.
