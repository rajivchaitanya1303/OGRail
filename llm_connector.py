import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

# def query_llm(prompt, model="meta-llama/llama-4-scout-17b-16e-instruct"):
def query_llm(prompt, model="meta-llama/llama-4-maverick-17b-128e-instruct"):
    try:
        response = client.chat.completions.create(
            model=model, 
            messages=[{"role": "user", "content": prompt}]
        )

        print("LLM Response:", response)

        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM Error: {str(e)}"