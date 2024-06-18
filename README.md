

## Environments

CUDA 11.3
torch 1.12.3

## Installation

We train and test based on Python3.9 and Pytorch. To install the dependencies run:

```
pip install -r requirements.txt
```

## Download
<!-- - Download the following hugging-face models and put them in the root directory 
  Facebook Speech2Text 2
  sentence-transformers/all-mpnet-base-v2 -->
- Download video demo from [google-drive](https://drive.google.com/file/d/1BDSvBSeFuXv-b7lOXAN8NlIB-3pjGASv/view?usp=drive_link) and put it in `./videos/`
## Inference
- Before inference, decide which llm is chosen (e.g. if llama3 is to be used)
  `ollama pull llama3`
- Run the demo:
  `python ./rag_with_time_stamp.py --question "When is the man doing a break move?"`
