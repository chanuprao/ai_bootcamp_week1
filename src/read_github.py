import os
from openai import OpenAI

# Get API key from environment variable
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

# System & User Messages
system_prompt = "You are a Generalist with good General knowledge especially in geo locations"
user_prompt = "What country has the same letter repeated the most in its name?"

# GPT call (deterministic output)
response = client.chat.completions.create(
    model='gpt-4o',  # or 'gpt-4'
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0,  # make it deterministic
)

# Print the response
print("Prompt:", user_prompt)
print("Response:", response.choices[0].message.content)
