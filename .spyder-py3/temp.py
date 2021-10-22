import nltk
import re

paragraph = """ An IDE (Integrated Development Environment) is used for software development. An IDE may have a compiler, debugger, and all the other requirements needed for software development. IDEs help in consolidating different aspects of a computer program. IDE is also used for development in Data Science (DS) and Machine Learning (ML) due to its vast libraries.

Various aspects of code writing can be implemented through IDEs like compiling, debugging, building executables, editing source code, etc. Python is a widely used language by coders, and python IDEs help in coding & compiling easily. There are IDEs which are used a lot nowadays, let us see some of the best Python IDEs for DS & ML in the market."""

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wordnet = WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)
words = nltk.word_tokenize(paragraph)
corpus=[]
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('English'))]
    review = ' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
x= cv.fit_transform(corpus).toarray()  