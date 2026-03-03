import os
import pickle
import sys
os.system('pip install pyrender')
os.system('pip install pyglet==1.4.0a1')
os.system('pip install triangle==20220202')

import gradio as gr
from collections import Counter
import torch
import numpy as np
import pytorch_lightning as pl
import subprocess
from pathlib import Path
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.config import parse_args
from huggingface_hub import snapshot_download
import socket
import numpy as np
from collections import deque
import nltk
from nltk.tokenize import sent_tokenize

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

        cfg = parse_args(phase="webui") 
        cfg.FOLDER = 'cache'

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
        print("MODEL SETUP")



    def connect_to_socket(self):
        HOST = '127.0.0.1'  # The server's hostname or IP address
        PORT = 27016        # The port used by the server
        
        print("trying to connect to socket")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))

            print("Connected to C++ application")

            ct=0
            old_frame = []
            while True:
                data = s.recv(1024).decode('utf-8')
                print(data)
                if not data:
                    self.load_and_map_data(self.frame_data, False)
                    break 
                
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

    def receive_exact(self, sock, size):
        data = b''
        while len(data) < size:
            remaining = size - len(data)
            data += sock.recv(min(remaining, 4096))
        return data

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
        # np.save('./record_temp.pt', updated_data)
        converted_data = convert_and_return(updated_data)
        curr_tokens = self.load_motion(converted_data, repeat=True)

        torch.save(curr_tokens, './record_temp.pt')

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

if __name__ == '__main__':
    nltk.download('punkt')

    tokenizer = StreamTokenize()
    print("initialized--------------")
    tokenizer.connect_to_socket()
