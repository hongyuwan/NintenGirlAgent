# coding=utf-8
#
# Installation instructions for pyaudio:
# APPLE Mac OS X
#   brew install portaudio
#   pip install pyaudio
# Debian/Ubuntu
#   sudo apt-get install python-pyaudio python3-pyaudio
#   or
#   pip install pyaudio
# CentOS
#   sudo yum install -y portaudio portaudio-devel && pip install pyaudio
# Microsoft Windows
#   python -m pip install pyaudio
import os
from dotenv import load_dotenv

# 加载 .env 文件中的所有环境变量
load_dotenv()

# 从环境变量中获取 API_KEY
api_key = os.getenv("AliYun-API-key")

if api_key:
    print("成功加载 API key:", api_key)
else:
    print("未检测到 API key，请检查 .env 文件配置")

import time
import pyaudio
import dashscope
from dashscope.api_entities.dashscope_response import SpeechSynthesisResponse
from dashscope.audio.tts_v2 import *

# 将your-dashscope-api-key替换成您自己的API-KEY
dashscope.api_key = api_key
model = "cosyvoice-v1"
voice = "longxiaochun"


class Callback(ResultCallback):
    _player = None
    _stream = None

    def on_open(self):
        self.file = open("output.mp3", "wb")
        print("websocket is open.")

    def on_complete(self):
        print("speech synthesis task complete successfully.")

    def on_error(self, message: str):
        print(f"speech synthesis task failed, {message}")

    def on_close(self):
        print("websocket is closed.")
        self.file.close()

    def on_event(self, message):
        print(f"recv speech synthsis message {message}")

    def on_data(self, data: bytes) -> None:
        print("audio result length:", len(data))
        self.file.write(data)


callback = Callback()

synthesizer = SpeechSynthesizer(
    model=model,
    voice=voice,
    callback=callback,
)

synthesizer.call("'亲爱的, 我实话和你说，我<strong>非常想你</strong>，我给你讲个[breath]<strong>卖核弹的小女孩的故事</strong>.")
print('requestId: ', synthesizer.get_last_request_id())

from pydub import AudioSegment
import pyaudio
import io

# 读取 MP3 文件
audio = AudioSegment.from_file("output.mp3", format="mp3")

# 转换为 PCM 数据
pcm_data = audio.raw_data

# 初始化 PyAudio
p = pyaudio.PyAudio()

# 打开音频流
stream = p.open(
    format=p.get_format_from_width(audio.sample_width),
    channels=audio.channels,
    rate=audio.frame_rate,
    output=True
)

# 播放音频
stream.write(pcm_data)

# 清理资源
stream.stop_stream()
stream.close()
p.terminate()