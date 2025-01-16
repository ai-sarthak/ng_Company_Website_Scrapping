import nltk
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import random
import google.generativeai as genai

# Download NLTK data (if not already downloaded)
nltk.download('punkt')

# Function to limit text based on word count
def limit_text_by_word_count(text, max_words):
    words = nltk.word_tokenize(text)
    if len(words) > max_words:
        return ' '.join(words[:max_words])
    return text

# Your prompt template
template = """You are an expert in analyzing company websites to extract valuable information that will be used to create hyper-personalized emails for sales and marketing purposes. Your analysis will focus on key business aspects that can be leveraged to write effective emails, with the goal of improving deal closure rates and establishing strong business relationships.
    Please extract and structure the relevant information from the following text, making sure to include specific details about the company's operations, products, market focus, and any key points that could be used for personalized outreach.

    ### Scraped Text:
    {text}

    ---

    ### Please provide the extracted information in the following structured format:

    1. Company Overview
    2. Products/Services
    3. Target Audience/Market
    4. Key Business Initiatives
    5. Company Leadership & Team
    6. Industry Position
    7. Technology & Innovation
    8. Financial Information
    9. Recent News & Press Releases
    10. Challenges or Pain Points
    11. Opportunities for Engagement
    ---

    ### The goal is to ensure the information is structured in a way that helps craft personalized outreach, addressing the company's unique challenges and goals, and showcasing how we can provide value in a relevant and meaningful way.
    """

load_dotenv()

api_key1 = os.getenv('API_KEY1')
#api_key2 = os.getenv('API_KEY2')
#api_key3 = os.getenv('API_KEY3')
#API_Vault = [api_key1, api_key2, api_key3]
API_Vault = [api_key1]
genai.configure(api_key=random.choice(API_Vault))

# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

def Analyze_scrap(text):
    # Limit the input text to a maximum of 2000 words or adjust as needed
    max_input_words = 50000
    limited_text = limit_text_by_word_count(text, max_input_words)

    chat_session = model.start_chat(
        history=[]
    )

    # Use the limited text in the prompt
    prompt_text = template.format(text=limited_text)
    
    # Send the prompt to the model
    response = chat_session.send_message(prompt_text)

    return response.text
