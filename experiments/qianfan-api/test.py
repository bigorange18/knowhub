import qianfan
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv()  

def gen_wenxin_messages(prompt):
    '''
    构造文心模型请求参数 messages

    请求参数：
        prompt: 对应的用户提示词
    '''
    messages = [{"role": "user", "content": prompt}]
    return messages


def get_completion(prompt, model="ERNIE-Bot", temperature=0.01):
    '''
    获取文心模型调用结果

    请求参数：
        prompt: 对应的提示词
        model: 调用的模型，默认为 ERNIE-Bot，也可以按需选择 ERNIE-Bot-4 等其他模型
        temperature: 模型输出的温度系数，控制输出的随机程度，取值范围是 0~1.0，且不能设置为 0。温度系数越低，输出内容越一致。
    '''

    chat_comp = qianfan.ChatCompletion()
    message = gen_wenxin_messages(prompt)

    resp = chat_comp.do(messages=message, 
                        model=model,
                        temperature = temperature,
                        system="你是一名个人助理-小鲸鱼")

    return resp["result"]


import requests
import json

def wenxin_embedding(text: str):
    # 获取环境变量 wenxin_api_key、wenxin_secret_key
    api_key = os.environ['QIANFAN_AK']
    secret_key = os.environ['QIANFAN_SK']

    # 使用API Key、Secret Key向https://aip.baidubce.com/oauth/2.0/token 获取Access token
    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={0}&client_secret={1}".format(api_key, secret_key)
    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    
    # 通过获取的Access token 来embedding text
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/embedding-v1?access_token=" + str(response.json().get("access_token"))
    input = []
    input.append(text)
    payload = json.dumps({
        "input": input
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return json.loads(response.text)
# text应为List(string)
text = "今天天气怎么样。"
response = wenxin_embedding(text=text)
print(response)
