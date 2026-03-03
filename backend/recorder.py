import argparse
import sys
import tempfile
import subprocess
import os
import time

import numpy as np
import torch

def read_file(path):
    """Read content from a file."""
    with open(path, 'r') as file:
        return file.read()

def write_file(path, content):
    """Write content to a file."""
    with open(path, 'w') as file:
        file.write(content)

def run_script():
    """Run another Python script and wait for it to finish."""
    python_path = "\"C:\\Users\\Venkatasai Gudisa\\Desktop\\hexd\\MotionGPT\\record.py\""
    print(python_path)
    os.system("python " + python_path)

def main():
    # Parse command-line arguments
    file_path = sys.argv[1]

   
    # Call another Python script and wait for it to finish
    print("Calling another Python script...")
    run_script()

    temp_file_path = "C:\\Users\\Venkatasai Gudisa\\Desktop\\hexd\MotionGPT\\record_temp.pt"

    # Read the modified content from the temporary file
    modified_content = torch.load(temp_file_path)
    
    print(f"Read modified content from {temp_file_path}")

    print(file_path)
    # Write the modified content back to the original file
    torch.save( modified_content, file_path)
    print(f"Wrote modified content back to {file_path}")

    print(f"Deleted temporary file: {temp_file_path}")

if __name__ == "__main__":
    main()