import heapq
import json
import pickle
import socket
import collections
# from streamtokenize import StreamTokenize
import os
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
import speech_recognition as sr
import win32com.client
import struct
import math
import time

timestamp_sentence_pairs = []
curr_motion_ct = 1
start_time = time.time()

# Function to save a timestamp and sentence as a JSON pair
def save_timestamp_sentence(sentence, has_motion=False):
    # Get the current timestamp
    current_time = time.time()
    
    if has_motion:
    # Create a dictionary for the pair
        pair = {
            "timestamp": current_time - start_time,
            "sentence": sentence,
            "csv_file_num": curr_motion_ct,
        }
    else:
        pair = {
            "timestamp": current_time - start_time,
            "sentence": sentence,
        }
    
    # Add the pair to the list
    timestamp_sentence_pairs.append(pair)
    print(f"Saved: {pair}")

# Function to save all pairs to a JSON file
def save_to_json(filename="timestamp_sentences.json"):
    with open(filename, "w") as f:
        json.dump(timestamp_sentence_pairs, f, indent=4)
    print(f"All pairs saved to {filename}")

THRESHOLD = 0.5
# TODO: change to the actual max frame size of a recorded motion
MAX_FRAME_SIZE = 100

# TODO: the motions will be a mapping from the idx of the text to the motion
powerpoint = win32com.client.Dispatch("PowerPoint.Application")

def get_sentence_embedding(text):
    return model.encode(text)

def gaussian_similarity(x, y, sigma=10000.0):
    return math.exp(-(x - y)**2 / (2 * sigma**2))

def scaled_similarity(x,y):
    diff = abs(x - y)

    diff_score = (diff ** 2 / max(x, y) ** 2)
    return 1- diff_score

def receive_exact(sock, size):
    data = b''
    while len(data) < size:
        remaining = size - len(data)
        data += sock.recv(min(remaining, 4096))
    return data

def get_motion_sim(sender, pre_recorded_motion):
    get_message = 'GET_FRAMES'
    # send message thru socket for getting motion tokens of most recent motion
    sender.sendall(get_message.encode('utf-8'))
    print('sender sent successfully')

    # Usage
    header = receive_exact(sender, 4)
    data_size = struct.unpack("!I", header)[0]
    received_data = receive_exact(sender, data_size)
    live_motion = pickle.loads(received_data)
    #print("-----LIVE MOTION TOKENS----", live_motion)

    t1 = live_motion.float().flatten()
    t2 = pre_recorded_motion.float().flatten()


    # mask1 = (t1 != 462) 
    # mask2 = (t2 != 462)
        # mask1 = t1
        # mask2 = t2

    # t1_masked = t1*mask1
    # t2_masked = t2*mask2

    t1_masked = t1
    t2_masked = t2

    print(t1_masked)
    print(t2_masked)

    sum1 = t1_masked.sum()
    sum2 = t2_masked.sum()

    print(sum1)
    print(sum2)


    sim_score = scaled_similarity(sum1, sum2)

    print(sim_score)
    
    return sim_score

def find_top_similar_texts(target_text, file_embeddings, texts, n=3):
    target_encoding = model.encode(target_text, convert_to_tensor=True)

    print(target_encoding.shape, file_embeddings[texts[0]].shape)
    embeddings_list = [(text, F.cosine_similarity(target_encoding.unsqueeze(0), file_embeddings[text].unsqueeze(0), dim=1).item()) for text in texts]
    print(embeddings_list)
    sorted_embeddings_list = sorted(embeddings_list, key=lambda x: x[1])

    scores = []
    for text in texts:
        scores.append((text, F.cosine_similarity(target_encoding.unsqueeze(0), file_embeddings[text].unsqueeze(0), dim=1)))
    
    print("------------CURRENT SCORES ARE", scores)
    
    return sorted_embeddings_list[-n:]

def init_socket():
    # Connect to video process
    sender = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sender.connect(('127.0.0.1', 5000))
    print("Connected to sender socket")
    return sender

if __name__ == "__main__":
    # tokenize_obj = StreamTokenize()
    # tokenize_obj.print_motion_string("C:\\Users\\Venkatasai Gudisa\\Desktop\\hexd\\MotionGPT\\data\\HumanML3D\\HumanML3D\\new_joints_from_recs\\rotated_a2_jump1.npy", True)

    
    # live_motion_tokens = tokenize_obj.connect_to_socket()
    ### TODO:   start voice recording and tokenize each sentence as it comes in
    ###         check each token to every slide and if the similarity score is beyond a certain threshold,
    ###         trigger tokenize_obj.get_motion_accuracy(path of the slide's motion, motion threshold)
    ###         if that returns True, then display_slide(slide_path)

    ###            ['sent1', 'sent2',...]
    ###             [slide1, slide2, ...]
    ###             slide --> a motion token

    r = sr.Recognizer()
    presentation = powerpoint.ActivePresentation
    slideshow =  presentation.SlideShowSettings.Run()
    sender = init_socket()

    def go_to_slide(slide_num):
        slideshow.View.GotoSlide(int(slide_num))
        print(slide_num)

    with sr.Microphone() as source:
        # Adjust for ambient noise
        r.adjust_for_ambient_noise(source)

        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # stores a list of embeddings per file corresponding to the sentences in each file
        file_embeddings = collections.defaultdict(list)
        i=0
        
        json_file_path = "./animation_data.json"  # Replace with the path to your JSON file
        with open(json_file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        # Dictionary to store embeddings
        file_embeddings = {}
        file_idx = {}
        curr_idx = 0
        texts=[]

        # Loop through each element in the JSON data
        for item in json_data:
            # Extract the target text
            target_text = item.get("text")
            if not target_text:
                continue  # Skip if target_text is missing

            # Encode the target text
            embeddings = model.encode(target_text, convert_to_tensor=True)
            
            texts.append(target_text)
            # Store the embeddings and index
            file_embeddings[target_text] = embeddings
            file_idx[target_text] = curr_idx
            curr_idx += 1
        
        print('Loaded all of the slide texts and computed their embeddings')
        curr_idx = 0

        motion_tokens = {}

        # Loop through each element in the JSON data
        for item in json_data:
            # Check if the element has a "motion_path" field
            motion_num = item.get("motion_num")
            if not motion_num:
                continue
                
            motion_path = f"C:\\Users\Venkatasai Gudisa\\Desktop\\hexd\\AnimationOverlay\\ui\\motions\\navigation\\curr_motion_{motion_num}.pt"
            motion_tokens[motion_num] = torch.load(motion_path)
        
        print('Loaded all of the slide motions and computed their sums')
        
        try: 
            while True:
                try:
                    print("STARTED LISTENING")
                    
                    files_past_threshold = []
                    # Listen for a sentence
                    audio = r.listen(source, phrase_time_limit=5)  # 5 second limit per phrase
                    # Convert to text
                    target_text = r.recognize_google(audio)
                    print("Recognized:", target_text)

                    max_name, max_sim = '',0
                    top_sims = find_top_similar_texts(target_text, file_embeddings, texts)

                    total_sim_heap = []
                    
                    for curr_identifier, curr_sim in top_sims:
                        # print(curr_path, curr_sim)
                        if curr_sim < THRESHOLD:
                            continue

                        texts_with_motion = set()
                        for item in json_data:
                            if item.get("text") == curr_identifier:  # Replace "target_text" with the correct field name
                                curr_path = item.get("motion_num")
                                curr_slide = int(item.get("slide"))
                                if curr_path: # a motion exists for this slide
                                    # current_motion = tokenize_obj.get_current_motion_tokens(MAX_FRAME_SIZE)
                                    motion_sim_score = get_motion_sim(sender, motion_tokens[curr_path])

                                    # everytime we request a motion, we increment this counter so it matches up with the counter from streamtokenize
                                    curr_motion_ct += 1

                                    print('motion sim score for index', file_idx[curr_path], 'is',motion_sim_score)

                                    avg_sim = (motion_sim_score+curr_sim) / 2
                                    print(avg_sim)
                                    if avg_sim >= THRESHOLD:
                                        print('Slide appended to threshold breakers', curr_path)

                                        texts_with_motion.add(curr_identifier)
                                        heapq.heappush(total_sim_heap, (-avg_sim, curr_slide))
                                else:
                                    heapq.heappush(total_sim_heap, (-curr_sim, curr_slide))
                            # print(total_sim_heap)
                    # TODO: send signal to that particular slide
                    if total_sim_heap: 
                        # TODO: this is where we send a signal to display the animation/text
                        winner = heapq.heappop(total_sim_heap)[1]
                        go_to_slide(winner)
                    else:
                        print("No similar texts!")
                
                except sr.UnknownValueError:
                    print("Could not understand audio")
                except sr.RequestError:
                    print("Service error")
        finally:
            save_to_json()