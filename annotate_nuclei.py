import json
from vllm import LLM, SamplingParams
import requests
import pandas as pd

def creating_csv():
    documents = []
    with open("./data/MAVEN_ERE/train.jsonl", 'r') as f:
        data = [json.loads(line) for line in f]
    for d in data:
        documents.append(' '.join(d["sentences"]))
    df = pd.DataFrame({"text": documents})
    df.to_csv("./data/MAVEN_ERE/train_joint.csv", index=False)

    documents = []
    with open("./data/MAVEN_ERE/valid.jsonl", 'r') as f:
        data = [json.loads(line) for line in f]
    for d in data:
        documents.append(' '.join(d["sentences"]))
    df = pd.DataFrame({"text": documents})
    df.to_csv("./data/MAVEN_ERE/valid_joint.csv", index=False)


def annotate():
    llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", tensor_parallel_size=2)
    sampling_params = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=500)
    file = open(f"prompt.txt", "r")
    prompt = file.read()
    # df = pd.read_csv("./data/MAVEN_ERE/train_joint.csv")
    # prompts = [f"{prompt}\n{snippet}\n\n[Paste the document here, add the tags and remove the brackets]" for snippet in df["text"].values]
    # outputs = llm.generate(prompts, sampling_params)
    # generated_texts = [output.outputs[0].text for output in outputs]
    # print(generated_texts[:5])
    # df["annotated_text"] = generated_texts
    # df.to_csv("./data/MAVEN_ERE/train_annotated.csv", index=False)

    df = pd.read_csv("./data/MAVEN_ERE/valid_joint.csv")
    prompts = [f"{prompt}\n{snippet}\n\n[Paste the document here, add the tags and remove the brackets]" for snippet in df["text"].values]
    outputs = llm.generate(prompts, sampling_params)
    generated_texts = [output.outputs[0].text for output in outputs]
    df["annotated_text"] = generated_texts
    df.to_csv("./data/MAVEN_ERE/valid_annotated.csv", index=False)


if __name__ == "__main__":
    creating_csv()
    annotate()