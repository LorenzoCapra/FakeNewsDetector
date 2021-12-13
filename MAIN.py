import pickle

# Build a user-friendly program:
print('Hello! Thanks to this program you can identify fake news.\n')

title = input('Please, copy and paste here the title of the article to control:\n')
author = input('\n\nPlease, report here the author of the article to control:\n')
text = input('\n\nPlease, copy and paste here the text of the article to control:\n')

# Load the model:
model_tf = pickle.load(open("model_tf.pickle", "rb"))
tf = pickle.load(open("vectorizer.pickle", "rb"))

Xtf = tf.transform([text + title + author])

# Predict the label for the input article:
prediction = model_tf.predict(Xtf)
if prediction == 0:
    print('\nThe article is a reliable source')
else:
    print('\nWarning!! The article is a fake news!')
