import streamlit as st
import json
from Intelligent_chatbot import predict_class, get_response, analyze_sentiment, recognize_entities

st.title("Intelligent Therapeutic Chatbot")

# Load the intents from the intents.json file
with open('intents.json', 'r') as file:
    data = json.load(file)

def main(intents):
    st.text("Enter your message:")
    message = st.text_input("", "")
    if st.button("Send"):
        return_list = predict_class(message)
        response = get_response(return_list, data_json=data)  # Pass intents dictionary
        st.text_area("Bot's Response:", response, height=200)

        # Perform sentiment analysis
        sentiment = analyze_sentiment(message)
        st.text(f"Sentiment: {sentiment}")

        # Perform entity recognition
        entities = recognize_entities(message)
        st.text(f"Entities: {entities}")

# Call the main function with the loaded intents dictionary
main(data)