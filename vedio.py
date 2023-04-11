import bs4 as bs #beautiful Soup --> pulls data from html/xml files
import urllib.request
import re
import nltk

#installing libraries
#importing dependencies
import subprocess 
from ibm_watson import SpeechToTextV1
from ibm_watson.websocket import RecognizeCallback, AudioSource
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator 
#authenticate speech to text service 

#AI --> Q&A,PREDICTIONS,AUTOMATE COMPLEX PROCESSES
#!pip install ffmpeg

#command ='ffmpeg -i aiml.mkv -ab 160k -ar 44100 -vn aiml.wav'   #FFmpeg --> transcoding, streaming etc .. (.mkv, .flv, and .mov.)
#subprocess.call(command, shell=True)

#setup STT services
apikey = 'pFtpG-uagKwRJLhP4TosJMw14-LIWq9Ok9fzPQZKAEE9'
url = 'https://api.au-syd.speech-to-text.watson.cloud.ibm.com/instances/e1018c30-9f10-47d9-8f40-f048c60d0376'

# Setup service
authenticator = IAMAuthenticator(apikey)
stt = SpeechToTextV1(authenticator=authenticator)
stt.set_service_url(url)

#open audio source and convert 
with open('aiml.wav', 'rb') as f:
    res = stt.recognize(audio=f, content_type='audio/wav', model='en-AU_NarrowbandModel').get_result()
res.keys()

len(res['results'])

text = [result['alternatives'][0]['transcript'].rstrip() + '.\n' for result in res['results']]
text = [para[0].title() + para[1:] for para in text]
transcript = ''.join(text)
with open('aiml_converted.txt', 'w') as out:
    out.writelines(transcript)

text
transcript
len(text)
text = [para[0].title() + para[1:] for para in text]
transcript = ''.join(text)
with open('aiml_converted.txt','w') as out:
    out.writelines(transcript)

import nltk
nltk.download('stopwords')

import nltk #language processing
nltk.download('punkt')

topwords = nltk.corpus.stopwords.words('english')

word_frequencies = {}
for word in nltk.word_tokenize(transcript):
    if word not in stopwords:
        if word not in word_frequencies.keys():
            word_frequencies[word] = 1
        else:
            word_frequencies[word] += 1 

word_frequencies

maximum_frequency = max(word_frequencies.values())

for word in word_frequencies.keys():
    word_frequencies[word] = (word_frequencies[word]/maximum_frequency)

maximum_frequency

word_frequencies

sentence_scores = {}
for sent in text:
    for word in nltk.word_tokenize(sent.lower()):
        if word in word_frequencies.keys():
            if len(sent.split(' ')) < 100:
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word]
                else:
                    sentence_scores[sent] += word_frequencies[word]

sentence_scores

import heapq
summary_sentences = heapq.nlargest(10, sentence_scores, key = sentence_scores.get)

summary = ' '.join(summary_sentences)
print(summary)

with open('summarised_text.txt','w') as out:
    out.writelines(summary)

