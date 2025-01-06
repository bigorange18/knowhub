import os
import openai
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]

def create_prompt():
    ...
def main():
    ...

if __name__ == "__main__":
    print(openai.api_key)