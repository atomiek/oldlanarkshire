#!/bin/bash


#download requirements
echo "Updating package lists.."
sudo apt-get update -y

#install system dependencies
echo "Installing build-essential, cmake, wget, and Python headers.."
sudo apt-get install -y build-essential cmake python3-dev python3-venv wget


echo "Unsetting CXX so Cmake doesn't throw some errors.."
unset CXX
export CXX=""

#delete .venv if it currently exists
rm -r .venv
#create new ..venv
python3 -m venv .venv
echo "Activating python virtual environment..."
source ".venv/bin/activate"

.venv/bin/pip install --upgrade pip

if [ ! -f ".venv/bin/pip" ]; then
  echo "Pip path incorrect or doesnt exist"
  exit 1
else
  .venv/bin/pip install -r ./backend/requirements.txt
fi

# Download model if not present
MODEL_DIR="./backend/models"

MODEL_URL="https://huggingface.co/MaziyarPanahi/SciPhi-Self-RAG-Mistral-7B-32k-Mistral-7B-Instruct-v0.1-GGUF/resolve/main/SciPhi-Self-RAG-Mistral-7B-32k-Mistral-7B-Instruct-v0.1.Q8_0.gguf?download=true"
MODEL_NAME="SciPhi-Self-RAG-Mistral-7B-32k-Mistral-7B-Instruct-v0.1.Q8_0.gguf"

# Create models directory
mkdir -p "$MODEL_DIR"

# Check if model exists
if [ ! -f "$MODEL_DIR/$MODEL_NAME" ]; then
    echo "Downloading model..."
    wget "$MODEL_URL" -O "$MODEL_DIR/$MODEL_NAME"
    echo "Model downloaded to $MODEL_DIR/$MODEL_NAME"
else
    echo "Model already exists at $MODEL_DIR/$MODEL_NAME"
fi
