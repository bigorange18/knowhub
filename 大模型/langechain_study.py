import getpass
import os


from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"



def main():
    model = ChatOpenAI()
    model.invoke("your name is?")
    pass

if __name__ == "__main__":
    main()