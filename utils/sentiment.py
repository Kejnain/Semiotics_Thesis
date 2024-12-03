import os
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn import set_config

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

def processText(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)

parent = os.path.abspath(os.path.join(os.getcwd(), '..'))
data = 'data'
path = os.path.join(parent, data, 'goemotions.csv')
read = pd.read_csv(path)
features = read['text'].apply(processText)
labels = read.iloc[:, 9:]

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels.iloc[:, 0]
)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        strip_accents='unicode',
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True
    )),
    ('scaler', StandardScaler(with_mean=False)),
    ('classifier', MultiOutputClassifier(RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced_subsample',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )))
])

set_config(display='diagram')
pipeline

param_grid = {
    'tfidf__max_features': [10000, 15000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'classifier__estimator__n_estimators': [100, 200],
    'classifier__estimator__max_depth': [15, 20],
    'classifier__estimator__min_samples_split': [2, 5],
    'classifier__estimator__min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2,
    scoring='accuracy',
    error_score='raise'
)

history = grid_search.fit(X_train, y_train)