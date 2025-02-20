import os
from google import genai
from dotenv import load_dotenv

load_dotenv(os.path.join(".", ".env"))
GEMINI_API_KEY = os.getenv("GENAI_API_KEY")

client = None
if GEMINI_API_KEY:
  try:
    client = genai.Client(api_key=GEMINI_API_KEY)
  except:
    pass

examples = {
  "tony": "Tony has 5 Apples. His friend Anthony has another 3 Apples. His mother asked him to collect a sum of 10 Apples before returning home. How many Apples are left for Tony to collect?",
  "gamer": "A gamer has 2 graphics cards in his computer. He buys two more graphics cards, but gifts one to a friend. How many graphics cards does he have?"
}


def is_gemini_available():
  return client is not None

def create_cot_zero_shot_prompt(user_prompt):
    return f"""
    Imagine your are an advanced AI Assistant that follows Chain of Thought (CoT) reasoning principle.
    You are asked to answer the following question:
    
    Question: {user_prompt}
    
    Let's think step by step:
    """

response = client.models.generate_content(
  model="gemini-1.5-pro", contents=examples["tony"]
)
print(response.text)




