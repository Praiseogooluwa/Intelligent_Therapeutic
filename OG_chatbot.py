import json #for loading json file
import nltk #for natural language processing
from nltk.stem import WordNetLemmatizer #for lemmatization 
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')
import numpy as np #for numerical computation in python
from tensorflow.keras import preprocessing #for building the models
import tensorflow as tf #used for deepl learning and machine learning tasks
import warnings
warnings.filterwarnings('ignore')
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation 
from tensorflow.keras.optimizers import Adam
import time
import pickle
from keras.models import load_model
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
import datetime



# Load the data from the JSON file
with open('intents.json', 'r') as file:
    data = json.load(file)

#Preprocess the data
lemmatizer = WordNetLemmatizer() #Initialize the lemmatizer
words = []
classes = []
documents = []
ignore=['?','!',',''.']

for intent in data['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        words_list = nltk.word_tokenize(pattern)
        words.extend(words_list)
        documents.append((words_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#Lemmatize, change each word case to lowercase and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore]
words = sorted(list(set(words)))

# Remove duplicates and sort classes
classes = sorted(list(set(classes)))

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))
# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(w.lower()) for w in pattern_words]
    # Create the bag of words array
    for w in words:
        bag.append(1 if w in pattern_words else 0)
    # Output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append(bag + output_row)  # Concatenate the bag of words and output row

# Convert the 'training' list into a NumPy array
np.random.shuffle(training)
training = np.array(training)
x_train = training[:, :len(words)]
y_train = training[:, len(words):]

#MODEL BUILDING 
model = Sequential()
model.add(Dense(128, input_shape=(len(x_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))

adam = Adam(learning_rate=0.001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

#fitting the model
weights = model.fit(np.array(x_train), np.array(y_train), epochs=200, batch_size=10, verbose=1)
#saving the model
model.save('chatbot_model.h5', weights)


#loading the models
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

#PREDICTION 
def clean_up(sentence):
    sentence_words=nltk.word_tokenize(sentence)
    sentence_words=[ lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence, words):
    sentence_words=clean_up(sentence)
    bag= [0] * len(words)
    
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow=bag_of_words(sentence,words)
    res=model.predict(np.array([bow]))[0]
    threshold=0.8
    results=[[i,r] for i,r in enumerate(res) if r> threshold]
    results.sort(key=lambda x: x[1],reverse=True)
    return_list=[]
    for result in results:
        return_list.append({'intent':classes[result[0]],'probability':str(result[1])})
    return return_list


def get_response(return_list,data_json):
    if len(return_list)==0:
        tag='no-answer'
    else:    
        tag=return_list[0]['intent']
    
    if tag == 'datetime':
        current_time = datetime.datetime.now().strftime("%A, %d %B %Y, %H:%M:%S")
        return current_time
    
    intent_list = data_json['intents']
    for intent in intent_list:
        if tag==intent['tag'] :
            result= np.random.choice(intent['responses'])
    return result

def response(text):
    return_list=predict_class(text)
    response=get_response(return_list, data)
    return response



# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Load the spaCy English model for entity recognition
nlp = spacy.load("en_core_web_sm")

# Function for sentiment analysis
def analyze_sentiment(text):
    scores = sid.polarity_scores(text)
    sentiment = "positive" if scores["compound"] >= 0 else "negative"
    return sentiment

# Function for entity recognition
def recognize_entities(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    return entities


# Function for generating a response based on sentiment and entities
def generate_response(sentiment, entities, intents_json):
    # Placeholder for the actual logic to generate a response based on sentiment and entities
    if sentiment == "positive" and "greeting" in entities:
        response = "Hello! How can I assist you today?"
    else:
        response = "I'm sorry to hear that. How can I help you feel better?"
    
    return response
