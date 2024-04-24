import random
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse, StreamingResponse
from starlette.websockets import WebSocketDisconnect
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import pandas as pd
import pyttsx3
from pathlib import Path

import speech_recognition as sr
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

#we declare a bunch of global variables that keep track of the conversation
app = FastAPI()
count = 0
array_images = []
start_input = ""

#necessary lines, it doesn't work without these lines
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/images", StaticFiles(directory="images"), name="images")
app.mount("/generative", StaticFiles(directory="generative"), name="generative")

#initialize computer microphone
engine = pyttsx3.init()

# Initialize OpenAI API KEY
client = OpenAI(
    api_key="OPENAI-API-KEY",
)

#dataset containing all the paintings
dataframe = pd.read_csv('dataset/paintings.csv')
#dataset containing images AI-generated
final_dataframe = pd.read_csv('generative/immagini_finali_ai.csv')

#function to transcribe audio into text
def whisper(audio):
    with open("speech.wav", "wb") as f:
        f.write(audio.get_wav_data())
        speech = open("speech.wav", "rb")
        #call to whisper model
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=speech,
            response_format="text"
        )
        print(transcription)
        return transcription

#function that returns the names of the 3 images that best match user audio input
def get_images(testo):
    global array_images

    template = """
        You are a CSV wizard able to extract information from a CSV file.
        The CSV {dataframe} has 8 columns. 
        You have ONLY to focus on the columns 'parola1', 'parola2', 'parola3'; their values need to pattern the user input {quest}.
        Your task is to find the perfect set of paths for the user input. Do return exactly 3 image paths. 
        Return ONLY the image paths as a unique list, like ['path/to/image1.jpg', 'path/to/image2.jpg'].
        """

    prompt = PromptTemplate(
        input_variables=['dataframe', 'quest', 'array_images'],
        template=template,
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=1, max_tokens=100)
    chain = LLMChain(llm=llm, prompt=prompt)
    quest = testo
    response = chain.run(dataframe=dataframe, quest=quest, array_images=array_images)
    array_paths = response.replace('[', '').replace(']', '').replace(" ", '').replace("'", '').split(',')
    return array_paths

#function that, given the 3 images of the room, decides which fits better to the audio prompt given by user
def chosen(last3, testo):
    desc = [dataframe[dataframe["image path"] == str(last3[i])]["description"].values[0] for i in range(3)]

    template = """
        You want to understand which one of the images {last3} fits the best 
        to the user input: {input}. You are a model that 
        Return ONLY the index of the image that fits best - so you have to give me 0 if you choose image 1,
        #1 if you choose image 2, 2 if you choose image 3.
        #If you do not understand and you are not able to choose, SILENTLY randomly choose a number from 0 to 2 and ONLY return THE NUMBER.
        """
    prompt = PromptTemplate(
        input_variables=['last3','input'],
        template=template,
    )
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8, max_tokens=100)
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(last3=last3, input=testo)
    if isinstance(response, str):
        response = random.randint(0, 2)
    return int(float(response))

#function that generates an audio response that takes the user to the next "room" of the museum
def create_video(last3, chosen_image, testo):
    global count
    global start_input

    descrip = dataframe[dataframe["image path"] == str(last3[chosen_image])]["description"].values[0]

    template = """
        You are a tourist guide for a an immersive journey of 4 steps regarding this content {context}. 
        The current step is number: {count}. So, e.g. if count is 1, you could introduce the journey in an engaging way while when count is equal to 4 you can conclude resume the whole experience.
        The current path is influenced by user feelings: {testo} experienced watching at the current image, that has as a description {descrip}.
        You have to return a short engaging story focusing on feelings.   
        Is mandatory to answer in Italian and don't use more than 35 words!
        """
    prompt = PromptTemplate(
        input_variables=['context', 'count', 'testo', 'descrip'],
        template=template,
    )
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8, max_tokens=100)
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(context=start_input, count=count, testo=testo, descrip=descrip)
    return response

def final_image():
    global start_input

    template = """
        You are a tourist guide for a an immersive journey of 4 steps regarding this content {context}. 
        The current step is the final. 
        Your task is to look at {final_dataframe} for the description that best fits the user context.
        You have to return JUST the image path of the best fitting image.
        If you are not sure about the result, SILENTLY return a random image path.
        """
    prompt = PromptTemplate(
        input_variables=['context', 'final_dataframe'],
        template=template,
    )
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8, max_tokens=100)
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(context=start_input,final_dataframe=final_dataframe)
    return response

#function that given the list of paintings returns the list of titles
def get_titles(lista):
    features = "titoli" + str([dataframe[dataframe['image path']==str(lista[i])]['title'].values[0] for i in range(len(lista))])
    return features

#function that given the list of paintings returns the list of years
def get_years(lista):
    features = "anni" + str([dataframe[dataframe['image path']==str(lista[i])]['year'].values[0] for i in range(len(lista))])
    return features

#function that given the list of paintings returns the list of respective authors
def get_authors(lista):
    features = "autori" + str([dataframe[dataframe['image path']==str(lista[i])]['author'].values[0] for i in range(len(lista))])
    return features

#main function that receive audio inputs and returns everything (images, audio, etc)
async def main_function(websocket: WebSocket):
    #call to the global variables, count keeps track of the number of steps
    global count
    global array_images
    global array_paths
    global start_input
    global frasi
    count += 1

    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    with microphone as source:
        #listen to the user prompt
        recognizer.adjust_for_ambient_noise(source, duration=0.5)  # set di default
        audio = recognizer.listen(source)
        try:
            #transcription of the prompt
            user_input = whisper(audio)

            #if we are in the first step we add the prompt to the conversation variable
            if count == 1:
                start_input += user_input

            #we get 3 images that match user input
            array_paths = get_images(user_input)

            #we keep track of the previous images in order not to show them again
            [array_images.append(array_paths[i]) for i in range(len(array_paths))]

            #if we aren't in the first step (we also want to return the audio "description")
            if count > 1:

                #we choose the "best" image among the 3 images
                chosen_img = chosen(array_images[-3:], user_input)

                #we generate the vocal output that is read by computer microphone
                generated = create_video(array_images[-3:], chosen_img, user_input)
                newVoiceRate = 145
                engine.setProperty('rate', newVoiceRate)
                engine.say(generated)
                engine.runAndWait()
                engine.stop()
                
                #we "tell" to the frontend which image has been selected
                await websocket.send_text(str(chosen_img))

            #if it's not the final step we send also the information relative to every painting
            if count <= 2:
                await websocket.send_text(array_paths)
                titles = get_titles(array_paths)
                await websocket.send_text(titles)
                years = get_years(array_paths)
                await websocket.send_text(years)
                authors = get_authors(array_paths)
                await websocket.send_text(authors)

            #if it's the last step, we only want to send the final images
            if count == 3:
                await websocket.send_text("ultimo")
                await websocket.send_text(array_paths)
                count, array_images, start_input = 0, [], ""


        except sr.UnknownValueError:
            print("No trascription available")  # No transcription
        except sr.RequestError as e:
            print(f"Error in the request: {e}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        try:
            # Receive messages from frontend
            message = await websocket.receive_text()

            if message == "startRecording":
                # If the conversation is started, we initialize the trascription
                await main_function(websocket)
            elif message == "endRecording":
                # Interrupt transcription and close websocket
                pass
        except WebSocketDisconnect:
            break 

    # End transcription when the socket is closed
    print("WebSocket connection closed")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    return FileResponse("templates/index.html")