import getpass
import os


from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
os.environ["LANGCHAIN_TRACING_V2"] = "true"



def main():
    model = ChatOpenAI()
    model.invoke("your name is?")
    pass

if __name__ == "__main__":
    main()