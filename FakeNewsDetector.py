import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Clean the dataset, removing all rows containing null information:
train_data = pd.read_csv('Data/train.csv').dropna()
test_data = pd.read_csv('Data/test.csv').dropna()  # We could use this data to predict new articles

# Separate features and labels, then split into train and test data:
X1 = train_data['text']
X2 = train_data['title']
X3 = train_data['author']
y = train_data['label']

X = X1 + X2 + X3

# Use a vectorizer to transform the text into a vector of numbers:
tf = TfidfVectorizer(stop_words='english', max_df=0.7, strip_accents='ascii', max_features=5000, lowercase=True)

Xtf = tf.fit_transform(X)

Xtf_train, Xtf_test, ytf_train, ytf_test = train_test_split(Xtf, y)

# Build the model, train it and make predictions:
model_tf = PassiveAggressiveClassifier()
model_tf.fit(Xtf_train, ytf_train)

# Save the model and the vectorizer:
pickle.dump(model_tf, open("model_tf.pickle", "wb"))
pickle.dump(tf, open("vectorizer.pickle", "wb"))

predictions_tf = model_tf.predict(Xtf_test)

# Compute the accuracy of the model:
print(f'Accuracy score (with Tfid Vectorizer): {accuracy_score(ytf_test, predictions_tf)}'
      f'\nPrecision score (with Tfid Vectorizer): {precision_score(ytf_test, predictions_tf)}'
      f'\nRecall score (with Tfid Vectorizer): {recall_score(ytf_test, predictions_tf)}')
