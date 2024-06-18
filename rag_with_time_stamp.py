import os
os.environ["CUDA_VISIBLE_DEVICES"]="6"
os.environ["TOKENIZERS_PARALLELISM"]="false"
import sys
from transformers import pipeline
from moviepy.editor import VideoFileClip
from datetime import datetime
import argparse
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import JSONLoader
import json
from langchain import hub
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

def metadate_func(record:dict, metadata:dict) -> dict:
    metadata["start_time"] = record.get("timestamp")[0]
    metadata["end_time"] = record.get("timestamp")[1]
    return metadata

def load_json_file(json_file):
    loader = JSONLoader(
    file_path=json_file,
    jq_schema=".results.xukun[]",
    content_key="sentence",
    metadata_func=metadate_func
    )
    return loader.load()


def split_text_encode_vectorstore(data, embedding_model, persist_directory, topk):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(data)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model, persist_directory=persist_directory)
    # retriever = vectorstore.as_retriever(search_kwargs={"k": topk}, search_type="similarity")
    retriever = vectorstore.as_retriever(search_type="mmr",
                search_kwargs={'k': topk, 'lambda_mult': 0.25})
    return retriever

def retrieve_docs(retriever, query):
    retrieved_docs = retriever.invoke(query)
    formatted_docs, start_time_list, end_time_list = format_docs(retrieved_docs)
    return formatted_docs, start_time_list, end_time_list

def format_docs(docs):
    result_text = []
    start_time_list = []
    end_time_list = []
    for doc in docs:
        start_time = doc.metadata["start_time"]
        end_time = doc.metadata["end_time"]
        page_content = doc.page_content
        temp_text = f"from {round(start_time, 2)}s to {round(end_time, 2)}s, {page_content}"
        result_text.append(temp_text)
        start_time_list.append(round(start_time, 2))
        end_time_list.append(round(end_time, 2))
    return "\n\n".join(result_text), start_time_list, end_time_list

def get_prompt(prompt, formatted_docs, question):
    prompt_temp = hub.pull(prompt)
    input_text = prompt_temp.invoke({"context":formatted_docs, "question":question})
    return input_text

def get_unrepeated_time(start_time_list, end_time_list):
    start_time_temp = []
    end_time_temp = []
    for i in range(len(start_time_list)):
        start_time = start_time_list[i]
        end_time = end_time_list[i]
        if (start_time in start_time_temp) and (end_time in end_time_temp):
            continue
        else:
            start_time_temp.append(start_time)
            end_time_temp.append(end_time)
    return start_time_temp, end_time_temp

def clip_video(video_path, output_path, start_time_list, end_time_list):
    start_time_list, end_time_list = get_unrepeated_time(start_time_list, end_time_list)
    video = VideoFileClip(video_path)
    now = str(datetime.now())[:-7].split(" ")
    root_dir = "_".join(now[0].split("-")+now[1].split(":"))
    root_dir = os.path.join(output_path, root_dir)
    os.makedirs(root_dir, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with open(os.devnull, 'w') as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        for i in range(len(start_time_list)):
            start_time = start_time_list[i]
            end_time = end_time_list[i]
            cut_video = video.subclip(start_time, end_time)
            video_name = str(start_time) + "_" + str(end_time) + ".mp4"
            output_video_path = os.path.join(root_dir, video_name)
            cut_video.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    return root_dir



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_model",
                        type=str,
                        # default="/data/wky/code/RAG/embedding_model/embedding_model")
                        default="sentence-transformers/all-mpnet-base-v2"
    parser.add_argument("--data_path",
                        type=str,
                        default = "/data/wky/code/RAG/dvc_results.json")
    parser.add_argument("--question",
                        type=str,
                        default = "When is the man doing a break dance?",)
                        # required=True)
    parser.add_argument("--vectorstore_path",
                        type=str,
                        default = "/data/wky/code/RAG/Chroma")
    parser.add_argument("--llm",
                        type=str,
                        choices=["llama2", "llama3"],
                        default="llama3")
    parser.add_argument("--video_path",
                        type=str,
                        default="/data/wky/code/RAG/videos/xukun.mp4")
    parser.add_argument("--retrieve_topk",
                        type=int,
                        default=1)
    parser.add_argument("--prompt",
                        type=str,
                        default="rlm/rag-prompt")
    parser.add_argument("--output_path",
                        type=str,
                        default="/data/wky/code/RAG/output")
    parser.add_argument("--audio_path",
                        type=str,
                        default=None)
    config = parser.parse_args()
# Preparation before the interaction

    embedding_model = HuggingFaceEmbeddings(model_name=config.embedding_model)
    data_file = config.data_path
    data = load_json_file(data_file)
    retriever = split_text_encode_vectorstore(data, embedding_model, config.vectorstore_path, config.retrieve_topk)

# Interaction Begin
    llm = Ollama(model=config.llm)
    print(f"Using {config.llm} in the following interaction.")
    if config.audio_path == None:
        question = config.question
    else:
        # asr = pipeline("automatic-speech-recognition", model="/data/wky/code/RAG/audio2text_model/audio2text_facebook", feature_extractor="/data/wky/code/RAG/audio2text_model/audio2text_facebook")
        asr = pipeline("automatic-speech-recognition", model="facebook/s2t-wav2vec2-large-en-de", feature_extractor="facebook/s2t-wav2vec2-large-en-de")

        question = asr(config.audio_path)["text"]
    formatted_docs, start_time_list, end_time_list = retrieve_docs(retriever, question)
    print(f"retrived docs is :\n{formatted_docs}")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with open(os.devnull, 'w') as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        output_dir = clip_video(video_path=config.video_path, output_path=config.output_path, start_time_list=start_time_list, end_time_list=end_time_list)
        input_text = get_prompt(config.prompt, formatted_docs, question)
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    print(f"Thinking................")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with open(os.devnull, 'w') as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        result = llm.invoke(input_text)
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    print(result)

    print(f"Video clips are saved in {output_dir}")


