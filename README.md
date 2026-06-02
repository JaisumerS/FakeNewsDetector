# Fake News Detection Classifier

## By: Jaisumer Sandhu, Alyan Qureshi, Jaival Patel
A machine learning project that classifies political and world news articles as **Real News** or **Fake News** using natural language processing, TF-IDF vectorization, and supervised learning models.

The project uses the Kaggle **Fake News Detection Dataset**, which includes separate CSV files for real and fake news articles. The model combines each article’s title and body text, converts the text into numerical features, trains multiple classifiers, and evaluates their performance using standard classification metrics.

## Dataset

Dataset: [Fake News Detection Dataset on Kaggle](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)

The dataset contains two CSV files:

- `True.csv` — real news articles
- `Fake.csv` — fake news articles

Each article includes:

- Title
- Text
- Subject
- Date

The project labels fake news as `1` and real news as `0`, then combines both datasets into one shuffled dataset for training and testing.

## Project Goals

The main goal of this project is to compare multiple machine learning models for fake news detection and evaluate which model performs best on high-dimensional text data.

The project focuses on:

- Processing real and fake news datasets
- Combining article titles and body text
- Converting text into TF-IDF features
- Training multiple classification models
- Comparing performance using validation metrics
- Reducing false negatives so fake news is less likely to be classified as real
- Allowing users to enter custom article text for prediction

## Models Used

### Multinomial Naive Bayes

Multinomial Naive Bayes is commonly used for text classification because it works well with word counts and term-frequency features. It is fast, lightweight, and effective as a baseline model for natural language processing tasks.

### Random Forest

Random Forest uses multiple decision trees to reduce variance and handle noisy data. It is useful for identifying important features and improving robustness, although it may not always perform best with sparse, high-dimensional text features.

### Linear Support Vector Classifier

LinearSVC is effective for high-dimensional text classification problems because it performs well with sparse TF-IDF features. It separates real and fake news articles by maximizing the decision boundary between classes.

## Machine Learning Workflow

### 1. Load Dataset

The dataset is downloaded using `kagglehub` and loaded into Pandas DataFrames.

```python
path = kagglehub.dataset_download("emineyetm/fake-news-detection-datasets")

base_path = os.path.join(path, "News _dataset")

fake_df = pd.read_csv(os.path.join(base_path, "Fake.csv"))
true_df = pd.read_csv(os.path.join(base_path, "True.csv"))
```

### 2. Validate Data

The project checks for missing values and verifies that each column contains consistent datatypes.

```python
print(fake_df.isnull().sum())
print(true_df.isnull().sum())
```

### 3. Label and Combine Data

Fake news articles are labelled as `1`, and real news articles are labelled as `0`.

```python
fake_df["label"] = 1
true_df["label"] = 0

df = pd.concat([fake_df, true_df], ignore_index=True)
df = df.sample(frac=1, random_state=0).reset_index(drop=True)
```

### 4. Combine Title and Article Text

The title and article body are combined into one input feature so the model can learn from both the headline and article content.

```python
df["combined_title_and_text"] = df["title"] + " " + df["text"]

X = df["combined_title_and_text"]
y = df["label"]
```

### 5. Split Data

The dataset is split into a 75% training set and 25% testing set.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=0
)
```

### 6. Convert Text with TF-IDF

The project uses `TfidfVectorizer` to convert article text into numerical features.

TF-IDF helps reduce the importance of common words and gives more weight to terms that are more meaningful for classification.

```python
TfidfVectorizer(
    stop_words="english",
    max_df=0.6,
    max_features=1000
)
```

### 7. Train Models

The project supports three models:

```text
1. LinearSVC
2. MultinomialNB
3. RandomForestClassifier
```

Each model is trained using a scikit-learn pipeline that combines TF-IDF vectorization with the selected classifier.

```python
Pipeline([
    ("tfidf", vectorizer),
    ("classifier", classifier)
])
```

## Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Precision-Recall AUC
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve

The confusion matrix was especially important because the project prioritized reducing false negatives. In this context, a false negative means a fake news article is incorrectly classified as real news.

## Hyperparameter Tuning

The project uses `GridSearchCV` to tune model parameters and improve performance.

Example SVM tuning parameters:

```python
param_grid = {
    "tfidf__max_df": [0.6, 0.7],
    "tfidf__max_features": [500, 1000],
    "classifier__C": [0.001, 0.01, 0.1]
}
```

The grid search uses 5-fold cross-validation and accuracy scoring.

```python
GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
```

## Results and Conclusions

The project compared Multinomial Naive Bayes, Random Forest, and LinearSVC on TF-IDF text features.

Key conclusions:

- **MultinomialNB** was fast, simple, and performed strongly on word-frequency-based features.
- **LinearSVC** worked well for high-dimensional sparse text data and produced strong classification results.
- **Random Forest** provided interpretability and robustness but was less ideal for sparse TF-IDF text features.
- The final comparison showed that MultinomialNB produced the strongest overall result for this dataset, followed by SVM, then Random Forest.
- The project prioritized reducing false negatives to limit fake news being classified as real.

## User Input Interface

The project includes a command-line interface where users can enter a news title and article body.

```text
Please enter the news title:
Please enter the news article text:
```

The title and article text are combined and passed into the trained model.

```python
combined = title.strip() + " " + article.strip()
full_article_pred = pipeline.predict([combined])
```

The program then returns a human-readable prediction:

```text
The full article is predicted to be: Fake News
```

or:

```text
The full article is predicted to be: True News
```

## Technologies Used

- Python
- Pandas
- scikit-learn
- TF-IDF Vectorizer
- Multinomial Naive Bayes
- Random Forest Classifier
- LinearSVC
- GridSearchCV
- Matplotlib
- Seaborn
- KaggleHub

## How to Run

### 1. Install dependencies

```bash
pip install pandas scikit-learn matplotlib seaborn kagglehub
```

### 2. Run the program

```bash
python main.py
```

If your project uses a different filename, replace `main.py` with the correct script name.

### 3. Select a model

When prompted, choose one of the available models:

```text
1. SVC Model
2. Multinomial NB
3. Random Forest
```

### 4. View metrics

The program will ask whether you want to view model metrics.

```text
Would you like to see the metrics of the model used? Enter y or n:
```

### 5. Test a custom article

Enter a news title and article text to receive a prediction.

## Project Structure

```text
fake-news-detection/
│
├── main.py
├── ML_models.py
├── README.md
└── requirements.txt
```

Suggested structure:

- `main.py` — loads data, trains the selected model, and runs predictions
- `ML_models.py` — contains model pipelines, input handling, and evaluation metrics
- `README.md` — project documentation
- `requirements.txt` — project dependencies

## Future Improvements

Future improvements could include:

- Adding lemmatization or stemming
- Testing n-gram TF-IDF features
- Saving trained models with `joblib`
- Building a Flask or FastAPI backend
- Creating a frontend for article submission
- Adding unit tests for preprocessing and model prediction
- Comparing transformer-based NLP models
- Adding support for larger and newer news datasets

## Disclaimer

This project is for educational purposes only. The model should not be used as the only source of truth for determining whether an article is real or fake. Real-world misinformation detection requires source verification, fact-checking, and human review.
