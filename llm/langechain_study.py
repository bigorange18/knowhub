import os
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import QianfanChatEndpoint

_ = load_dotenv(find_dotenv())  # 默认会加载根目录下的.env文件
os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890"
os.environ['HTTP_PROXY']  = "http://127.0.0.1:7890"



def create_prompt():
    define_prompt = ChatPromptTemplate.from_messages([
        ("system", "you are a assistant"),
        MessagesPlaceholder('msgs')
    ])
    res = define_prompt.invoke({
        "msgs": [
            HumanMessage(content="hello world")
        ]
    })
    return res

def main():
    chat = QianfanChatEndpoint(
        streaming=True,
    )
    res = chat([HumanMessage(content="讲一个故事")])
    print(res.content)


    # print(create_prompt())
    pass    

if __name__ == "__main__":
    main()