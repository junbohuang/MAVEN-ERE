import json
from vllm import LLM, SamplingParams
import requests
import pandas as pd
from itertools import chain
from openai import OpenAI
from tqdm import tqdm



def tag_mention_per_sentence(mention: dict, tokens: list[list]):
    tokenized_sentence = tokens[mention["sent_id"]]
    retrieved_mention = ' '.join(tokenized_sentence[mention["offset"][0]: mention["offset"][1]])
    assert retrieved_mention == mention["trigger_word"]
    tokenized_sentence[mention["offset"][0]] = "<event>"+tokenized_sentence[mention["offset"][0]]
    tokenized_sentence[mention["offset"][1]-1] = tokenized_sentence[mention["offset"][1]-1]+"</event>"
    return tokenized_sentence

def tag_mention_per_doc(data: dict):
    tagged_tokens = data["tokens"]
    mentions = [mention for event in data["events"] for mention in event["mention"]]
    for mention in mentions:
        tagged_tokens[mention["sent_id"]] = tag_mention_per_sentence(mention, tagged_tokens)
    return tokens_to_sentences(tagged_tokens)

def tokens_to_sentences(tagged_tokens):
    return ' '.join([' '.join(tagged_tokens_item) for tagged_tokens_item in tagged_tokens])


def creating_csv():
    documents = []
    with open("./data/MAVEN_ERE/train.jsonl", 'r') as f:
        data = [json.loads(line) for line in f]
    for d in data:
        documents.append(tag_mention_per_doc(d))
    df = pd.DataFrame({"text": documents})
    df.to_csv("./data/MAVEN_ERE/train_joint.csv", index=False)

    documents = []
    with open("./data/MAVEN_ERE/valid.jsonl", 'r') as f:
        data = [json.loads(line) for line in f]
    for d in data:
        documents.append(tag_mention_per_doc(d))
    df = pd.DataFrame({"text": documents})
    df.to_csv("./data/MAVEN_ERE/valid_joint.csv", index=False)


def annotate_with_gpt4():
    file = open(f"prompt.txt", "r")
    prompt = file.read()
    df = pd.read_csv("./data/MAVEN_ERE/train_joint.csv")
    prompts = [f"{prompt}\n{snippet}\n\n[Paste the document here; replace the tags and remove the brackets]" for snippet
               in df["text"].values]

    client = OpenAI()

    generated_text = []
    for prompt in tqdm(prompts):
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        generated_text.append(completion.choices[0].message)
    df["annotated_text"] = generated_text
    df.to_csv("./data/MAVEN_ERE/train_annotated.csv")



def annotate():
    llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", tensor_parallel_size=2)
    sampling_params = SamplingParams(temperature=1, top_p=1, max_tokens=256)
    file = open(f"prompt.txt", "r")
    prompt = file.read()
    # df = pd.read_csv("./data/MAVEN_ERE/train_joint.csv")
    # prompts = [f"{prompt}\n{snippet}\n\n[Paste the document here, replace the tags and remove the brackets]" for snippet in df["text"].values]
    # outputs = llm.generate(prompts, sampling_params)
    # generated_texts = [output.outputs[0].text for output in outputs]
    # print(generated_texts[:5])
    # df["annotated_text"] = generated_texts
    # df.to_csv("./data/MAVEN_ERE/train_annotated.csv", index=False)

    df = pd.read_csv("./data/MAVEN_ERE/valid_joint.csv")
    prompts = [f"{prompt}\n{snippet}\n\n[Paste the document here; replace the tags and remove the brackets]" for snippet in df["text"].values]
    outputs = llm.generate(prompts, sampling_params)
    generated_texts = [output.outputs[0].text for output in outputs]
    df["annotated_text"] = generated_texts
    df.to_csv("./data/MAVEN_ERE/valid_annotated.csv", index=False)


if __name__ == "__main__":
    creating_csv()
    # annotate()
    annotate_with_gpt4()