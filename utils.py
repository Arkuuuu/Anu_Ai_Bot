import os
import requests
import boto3
from groq import Groq

# Load API keys from .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = "us-east-1"

# ✅ Initialize AWS Polly client
polly_client = boto3.client(
    "polly",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

def speech_to_text(audio_data):
    """Convert speech to text using Groq's Whisper v3 Large model."""
    with open(audio_data, "rb") as audio_file:
        response = requests.post(
            "https://api.groq.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": audio_file},
            data={"model": "whisper-large-v3", "response_format": "text"},
        )
    return response.text.strip()

def text_to_speech(input_text):
    """Convert chatbot response to speech using AWS Polly."""
    response = polly_client.synthesize_speech(Text=input_text, OutputFormat="mp3", VoiceId="Joanna")
    with open("temp_audio_play.mp3", "wb") as f:
        f.write(response["AudioStream"].read())
    return "temp_audio_play.mp3"

def get_answer(messages):
    """Generate chatbot response using Groq API."""
    client = Groq(api_key=GROQ_API_KEY)
    system_message = [{"role": "system", "content": "You are a helpful AI chatbot, answering user queries."}]
    messages = system_message + messages

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )

    return response.choices[0].message.content
