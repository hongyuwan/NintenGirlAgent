import os
from dotenv import load_dotenv
import time
# 加载 .env 文件中的所有环境变量
load_dotenv()


api_key = os.getenv("AliYun-API-key")

if api_key:
    print("成功加载 API key:", api_key)
else:
    print("未检测到 API key，请检查 .env 文件配置")

# api_key = os.getenv("deepseek-API")
import os
from openai import OpenAI

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=api_key , # 如何获取API Key：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
#"https://dashscope.aliyuncs.com/compatible-mode/v1",
start_time = time.time()
def generateConversation(
    userName,
    conversationalUser,
    currentConversation,
    important_observations: list,
):
    prompt = {
        "information": {
            "Current conversation": currentConversation,
            "Conversation Memory":important_observations
        },
        "criteria": [
            f"Mark the words in the sentences that you feel you need to emphasize the emotion when you speak. Include one in each sentence, using<strong></strong>. Present one of emotion (Happy), (Sad), (Surprised), (Angry), (Fearful), (Disgusted), (Calm), (Serious) should happen there in the end of sentences in ().",
            f"Add [breath] in the sentence when you feel need to pause."
        ],
        "Example output": "To be honest[breath], it's <strong>too harmful</strong> to me.(Angry)",
    }
    return prompt
context={"context": f"I need you to play as a girl--Ninteimoto, having a lively response with Tony."}
prompt=generateConversation('Ninteimoto','Tony','I love you so much.','Tony is Shy. Tony loves playing basketball.')
print(prompt)
completion = client.chat.completions.create(
    model="deepseek-v3",
    messages=[
        {'role': 'system', 'content': str(context)},
        {'role': 'user', 'content': str(prompt)}
        ]
)
print(completion.choices[0].message.content)
end_time = time.time()

time=end_time - start_time
print(time)