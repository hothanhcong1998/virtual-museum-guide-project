from gtts import gTTS
from io import BytesIO
import os
import openai
#import speech_recognition as sr
#import pyaudio
import sys
from pydub import AudioSegment
from pydub.playback import play
from pydub.effects import speedup

def text_to_speech(text):
    if not text:
        return
 
    tts = gTTS(text, lang='en')
    tts.save("temp.mp3")
    print()
    print()
    print('VOXIE:')
    print(text)
    song = AudioSegment.from_mp3("temp.mp3")
    play(speedup(song,1.1,150))
    os.remove("temp.mp3")

# this function does not work because the problem of PyAudio on MacOS operation system
def speech_to_text(text):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Start speaking!")
        audio = r.listen(source)
        text = r.recognize_google(audio)
        print(f'You: {text}')
        
# inference the original GPT3
def davinci(prompt):
    model_name = "text-davinci-003"
    completion = openai.Completion.create(model=model_name, prompt=prompt, \
                                         temperature=0.7,max_tokens=256, \
                                         top_p=1, frequency_penalty=0, presence_penalty=0)
    return completion.choices[0]["text"]

# inference the fine-tuned GPT3
def finetune(prompt):
    model_name = "davinci:ft-personal:ml709-2023-05-02-09-16-50"
    completion = openai.Completion.create(model=model_name, prompt=prompt, \
                                         temperature=0.7,max_tokens=256, \
                                         top_p=1, frequency_penalty=0, presence_penalty=0, stop=[" ##END##"]) 
    return completion.choices[0]["text"]

