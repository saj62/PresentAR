import csv
import os
import pickle
os.system('pip install pyrender')
os.system('pip install pyglet==1.4.0a1')
os.system('pip install triangle==20220202')

import gradio as gr
from collections import Counter
import torch
import time
import numpy as np
import pytorch_lightning as pl
import subprocess
from pathlib import Path
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.config import parse_args
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
from huggingface_hub import snapshot_download
import socket
import numpy as np
from collections import deque
import struct
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util

from representation_convertor import convert_and_return



class StreamTokenize:
    def __init__(self):
        self.joint_mapping = {
            0: 0,
            18: 1,
            22: 2,
            1: 3,
            19: 4,
            23: 5,
            2: 6,
            20: 7,
            24: 8,
            2: 9,
            21: 10,
            25: 11,
            3: 12,
            4: 13,
            11: 14,
            26: 15,
            12: 17,
            5: 16,
            6: 18,
            13: 19,
            7: 20,
            14: 21,
        }

        cfg = parse_args(phase="webui")  # parse config file
        cfg.FOLDER = 'cache'
        # output_dir = Path(cfg.FOLDER)
        # output_dir.mkdir(parents=True, exist_ok=True)

        pl.seed_everything(cfg.SEED_VALUE)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        model_path = snapshot_download(repo_id="bill-jiang/MotionGPT-base")
        datamodule = build_data(cfg, phase="test")
        self.model = build_model(cfg, datamodule)
        state_dict = torch.load(f'{model_path}/motiongpt_s3_h3d.tar',
                                map_location="cpu")["state_dict"]
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.frame_data = list()
        self.sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("MODEL SETUP")

    def send_messages(host="127.0.0.1", port=5000):
        messages = ["Hello", "How are you?", "This is a test", "Goodbye"]

        time.sleep(1)  # Wait for the listener to start


    def connect_to_socket(self):
        HOST = '127.0.0.1'  # The server's hostname or IP address
        PORT = 27016        # The port used by the server
        
        print("trying to connect to socket")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            motion_ct = 1

            request_receiver = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            request_receiver.bind(('127.0.0.1', 5000))

            print("connected to camera, waiting for app.py")
            # listen for possible connections
            request_receiver.listen()

            conn, addr = request_receiver.accept()
            conn.setblocking(False)
            print(f"Connected to {addr}")

            print("Connected to C++ application")

            ct=0
            old_frame = []
            while True:
                data = s.recv(1024).decode('utf-8')
                print(data)
                # if not data:
                #     response = self.load_and_map_data(self.frame_data, False)
                #     break
                # Process the received data
                # print(data.strip())

                current_frame = old_frame + data.strip().split('\n')[:32] # splits the given frame by line, storing a list of strings where each string is a joint

                # print(current_frame, " ", ct)
                joints_list = []
                for joint in current_frame:
                    if len(joint.split(',')) != 4:
                        continue
                    joint_idx, pos_x, pos_y, pos_z = joint.split(',')
                    joints_list.append({'JointIndex':joint_idx, 'PosX':pos_x, 'PosY':pos_y, 'PosZ':pos_z})

                ct += 1
                self.frame_data.append(joints_list)
                print(len(self.frame_data))

                try:
                    message = conn.recv(1024)
                    print(message)
                    if message == b"GET_FRAMES" and len(self.frame_data) > 1:
                        # Send frames when requested
                        start = min(0, len(self.frame_data) - 1)
                        end = min(len(self.frame_data), start+100)

                        # print((self.frame_data)[start:end])
                        # print(self.frame_data)

                        # save the motion to be used as csv later on
                        with open('frame_data.csv', 'w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(['Frame', 'JointIndex', 'PosX', 'PosY', 'PosZ'])  # Write header
                            for frame_idx, joints in enumerate(self.frame_data):
                                for joint in joints:
                                    writer.writerow([frame_idx, joint['JointIndex'], joint['PosX'], joint['PosY'], joint['PosZ']])
                                    
                        motion_ct += 1
                        response = self.load_and_map_data(self.frame_data, False) #commented out since we're not using it

                        serialized_frames = pickle.dumps(response)
                        print("Tokens about to be sent have length:", len(serialized_frames))
                        # print("Tokens after being pickled and unpickled:", pickle.load(serialized_frames))
                        header = struct.pack("!I", len(serialized_frames))  # 4-byte network-order size
                        conn.sendall(header + serialized_frames)

                        print('FRAMES SENT!!!')

                        # self.compare_speech_to_sentences(conn, slide_texts, self.frame_data) #added

                except BlockingIOError:
                    continue

                if len(self.frame_data) > 512:
                    self.frame_data = self.frame_data[:256]

    def load_slide_texts(self):
        slide_texts = {}
        text_directory = './slides/slide_text'
        for filename in os.listdir(text_directory):
            file_path = os.path.join(text_directory, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    sentences = sent_tokenize(text)
                    slide_texts[filename] = sentences
        return slide_texts

    def compare_speech_to_sentences(self, conn, slide_texts, frame_data): #added frame_data

        try:
            message = conn.recv(1024)
            #print(message)
            if not message:
                print("connection broken")
                return

            # Unpack message
            target_text = pickle.loads(message)

            # target_text = "is the lord of the sky the bringer"  # Dummy value for testing
            print("recieved target_text:", target_text)
            best_slide = None
            max_similarity = -1
            best_sentence = None

            for filename, sentences in slide_texts.items():
                for sentence in sentences:
                    sentence_embedding = self.sentence_model.encode(sentence, convert_to_tensor=True)
                    target_embedding = self.sentence_model.encode(target_text, convert_to_tensor=True)
                    similarity = util.pytorch_cos_sim(sentence_embedding, target_embedding).item()

                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_slide = filename
                        best_sentence = sentence

            print(f"Best matching slide: {best_slide} with similarity: {max_similarity} and sentence: {best_sentence}")
            motion_tokens = None
            if len(frame_data) > 0 :
                motion_tokens = self.load_and_map_data(frame_data, False) #changed true to false # added the load_and_map_data
            else:
                print("No Frames available")

            return_data = (best_slide, max_similarity, motion_tokens)
            serialized_data = pickle.dumps(return_data)
            header = struct.pack("!I", len(serialized_data))
            conn.sendall(header + serialized_data)

        except Exception as e:
            print(f"An error occurred: {e}")




    def get_current_motion_tokens(self, frame_window_size):
        return self.load_and_map_data(frame_window_size)

    def get_motion_accuracy(self, pre_recorded_motion_path, threshold):
        curr_token_sum = self.load_and_map_data(100)
        pre_recorded_motion = torch.load(pre_recorded_motion_path)

        return abs(curr_token_sum - self.get_flattened_sum(pre_recorded_motion)) <= threshold

    def find_max_frequency(self, list_of_lists):
        # Flatten the list of lists and count occurrences
        flattened = [item[0] for item in list_of_lists]
        frequency = Counter(flattened)

        # Find the number with maximum frequency
        max_freq_number = max(frequency, key=frequency.get)

        return max_freq_number, frequency[max_freq_number]
    def receive_exact(self, sock, size):
        data = b''
        while len(data) < size:
            remaining = size - len(data)
            data += sock.recv(min(remaining, 4096))
        return data

    # def get_flattened_sum(self, tokens):
    #     return sum([num for num in tokens.flatten().numpy() if num != 499])

    def load_and_map_data(self, frames, get_string: False):

        updated_data= np.zeros((len(frames), 22, 3))
        curr_frame_num = 0
        for frame in frames:

            all_possible_joints = set()
            for joint in frame:
                if joint['JointIndex']: all_possible_joints.add(int(joint['JointIndex']))

            for i in range(32):
                if i not in all_possible_joints:
                    frame.append({'JointIndex': i, 'PosX': 0, 'PosY': 0, 'PosZ': 0})

            for joint in frame:
                assert len(joint.keys()) == 4, f"Number of joints isn't correct for joint {joint}!!"
                if not joint['JointIndex'] or int(joint['JointIndex']) not in self.joint_mapping:
                    continue
                new_joint_index = self.joint_mapping[int(joint['JointIndex'])]
                updated_data[curr_frame_num][new_joint_index] = [float(joint['PosX']), float(joint['PosY']),
                                                                  float(joint['PosZ'])]
            curr_frame_num += 1
        print("--------", updated_data)
        converted_data = convert_and_return(updated_data)
        curr_tokens = self.load_motion(converted_data, repeat=True)
        #print("CURRENT TOKENS ARE:________", curr_tokens, curr_tokens.shape)
        if get_string:
            motion_token_string = self.model.lm.generate_conditional(task='m2t', motion_tokens=curr_tokens,
                                                                     lengths=[curr_tokens.shape[0]])[0]
            return motion_token_string
        torch.save(curr_tokens, "current_tokens.pt")
        print("TOKENS SAVED!!")
        # token_sum = self.get_flattened_sum(curr_tokens)
        
        return curr_tokens
        # return abs(token_sum - 18260) <= 10000

    def load_motion(self, feats, convert=False, repeat=False):
        print("----------BEFORE CONVERSION SHAPE IS", feats.shape)
        if convert:
            feats = convert_and_return(feats)
        # print("----------PREVIOUS SHAPE WAS", torch.load("C:\\Users\\Venkatasai Gudisa\\Desktop\\hexd\\MotionGPT\\motion_results\\right_wave_1.pt").shape)
        # print("----------CONVERTED FEATS SHAPE IS", feats.shape)
        feats = torch.tensor(feats, device=self.model.device)
        feats = torch.tensor(feats, dtype=torch.float32)

        if len(feats.shape) == 2:
            # feats = feats.unsqueeze(-1).repeat(1,1,4).permute(0,2,1)
            feats = feats[None]

        if repeat:
            feats = feats.repeat((512 // feats.shape[0]) + 1, 1, 1)[:512, :, :]
        print("-------FINAL FEATS SHAPE RIGHT BEFORE TOKENIZING IS", feats.shape)
        # Motion tokens
        motion_lengths = feats.shape[0]
        motion_token, _ = self.model.vae.encode(feats)

        return motion_token

    def save_frame(self, file, data):
        np.save(file, data)

    def print_motion_string(self, motion_path, convert=False):
        print(motion_path)
        motion_token = self.load_motion(np.load(motion_path), convert)
        print("---------SHAPE IS", motion_token.shape)
        motion_token_string = self.model.lm.generate_conditional(task='m2t', motion_tokens=motion_token,
                                                                 lengths=[motion_token.shape[0]])[0]
        print("**************************MOTION STRING IS:", motion_token_string)


if __name__ == '__main__':
    nltk.download('punkt')
    tokenizer = StreamTokenize()
    tokenizer.connect_to_socket()
