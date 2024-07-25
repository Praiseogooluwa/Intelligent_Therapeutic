import streamlit as st
import json
from OG_chatbot import predict_class, get_response, analyze_sentiment, recognize_entities
import random


# Load spaCy model
#nlp = spacy.load("en_core_web_sm")

# Load the intents from the intents.json file
with open('intents.json', 'r') as file:
    data = json.load(file)

def get_response(return_list, data_json):
    # Default response if no intent is recognized
    result = "Sorry, I don't have an answer to that question, ask me something else."
    
    for intent in return_list:
        tag = intent['intent']
        for i in data_json['intents']:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
    return result

def main(intents):
    st.write("Enter your message:")
    message = st.text_input("", "")
    if st.button("Send"):
        return_list = predict_class(message)
        response = get_response(return_list, data_json=intents)  # Pass intents dictionary
        st.text_area("OG's Response:", response, height=200)

        # Perform sentiment analysis
        sentiment = analyze_sentiment(message)
        st.write(f"Sentiment: {sentiment}")

        # Perform entity recognition
        entities = recognize_entities(message)
        st.write(f"Entities: {entities}")

# Streamlit app header and title
logo1 = 'chatbot/praisebot.png'  # Ensure the correct path
st.set_page_config(page_title="OG Therapeutic Chatbot | By Praise Ogooluwa", page_icon=logo1, layout="wide")

st.write("# OG Therapeutic Chatbot :sunglasses:")
st.write("Made with love by - [Praise Ogooluwa Bakare](https://praiseogooluwa.github.io/)")
st.write("#### Welcome to OG Therapeutic Chatbot app! Type your message below:")

# Sidebar with social profiles and model parameters
st.sidebar.markdown("# Follow me on my Social Profiles")
st.sidebar.markdown(
    """<a href="https://github.com/Praiseogooluwa" target="_blank"><img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" alt="GitHub" width="60px"></a>
    <a href="www.linkedin.com/in/praise-ogooluwa" target="_blank"><img src="https://cdn1.iconfinder.com/data/icons/logotypes/32/circle-linkedin-512.png" alt="LinkedIn" width="60px"></a>
    <a href="https://www.facebook.com/Oshin2104?mibextid=ZbWKwL" target="_blank"><img src="https://cdn-icons-png.flaticon.com/128/5968/5968764.png" alt="Facebook" width="60px"></a>
    <a href="https://wa.me/message/QHJGXMYYBNVUJ1" target="_blank"><img src="https://cdn-icons-png.flaticon.com/128/15707/15707820.png" alt="Whatsapp" width="60px"></a>
    """,
    unsafe_allow_html=True,
)

# HTML sidebar to fine-tune model's parameters to customize the bot's responses.
st.sidebar.markdown("# Therapeutic Model Parameters")
emotional_support = st.sidebar.slider("Emotional Support", 0.0, 1.0, 0.7, 0.1)
response_depth = st.sidebar.number_input("Response Depth", 1, 5, 2, step=1)
non_judgmental_language = st.sidebar.slider("Non-Judgmental Language", 0.0, 1.0, 0.9, 0.1)
max_response_length = st.sidebar.number_input("Max Response Length", 50, 500, 200, step=50)
therapeutic_approach = st.sidebar.selectbox("Therapeutic Approach", ["Cognitive-Behavioral", "Humanistic", "Psychoanalytic"])
avoid_trigger_words = st.sidebar.checkbox("Avoid Trigger Words")

# Main app where user enters prompt and gets the response
user_input = st.text_area("You:", "", key="user_input")
generate_button = st.button("Generate Response")

# Chat history
chat_history = []
if generate_button and user_input.strip() != "":
    chat_history.append({"role": "user", "content": user_input})
    return_list = predict_class(user_input)
    response = get_response(return_list, data_json=data)
    chat_history.append({"role": "assistant", "content": response})

st.subheader("Chat History")
for message in chat_history:
    if message["role"] == "user":
        st.text_area("You:", value=message["content"], height=50, max_chars=200, key=f"user_history_{message['content']}", disabled=True)
    else:
        st.text_area("OG:", value=message["content"], height=500, key=f"chatbot_history_{message['content']}")

# Additional styling to make the app visually appealing
st.markdown(
    """
    <style>
        body {
            font-family: Montserrat, sans-serif;
        }
        .stTextInput>div>div>textarea {
            background-color: #f0f0f0;
            color: #000;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        .stTextArea>div>textarea {
            resize: none;
        }
        .st-subheader {
            margin-top: 20px;
            font-size: 16px;
        }
        .stTextArea>div>div>textarea {
            height: 100px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Run the main function
main(data)
