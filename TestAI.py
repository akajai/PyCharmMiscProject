#pip install langchain

from gtts import gTTS

text = "Hello, this is a sample MP3 file."
tts = gTTS(text=text, lang='en')
tts.save("sample.mp3")


