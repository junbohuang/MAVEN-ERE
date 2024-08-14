import json
from vllm import LLM, SamplingParams
import requests


def creating_csv():
    documents = []
    with open("./data/MAVEN_ERE/train.jsonl", 'r') as f:
        data = [json.loads(line) for line in f]
    for d in data:
        documents.append(' '.join(d["sentences"]))
    with open("./data/MAVEN_ERE/train_joint.txt", 'w') as f:
        for line in documents:
            f.write(f"{line}\n")

    documents = []
    with open("./data/MAVEN_ERE/valid.jsonl", 'r') as f:
        data = [json.loads(line) for line in f]
    for d in data:
        documents.append(' '.join(d["sentences"]))
    with open("./data/MAVEN_ERE/valid_joint.txt", 'w') as f:
        for line in documents:
            f.write(f"{line}\n")

def llama3_skynet_api(snippet, prompt):
    try:
        url = 'https://turbo.skynet.coypu.org/'
        request = requests.post(url, json={"messages": [{"role": "user",
                                           "content": f"{prompt}\n{snippet}"}],
                                "temperature": 0.1,
                                "max_new_tokens": 10}).json()
        return request.get("generated_text")
    except Exception as e:
        return e


def annotate():
    llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", tensor_parallel_size=2)
    sampling_params = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=500)
    file = open(f"prompt.txt", "r")
    prompt = file.read()
    with open("./data/MAVEN_ERE/train_joint.txt", encoding='utf-8') as f:
        data = f.readlines()
    prompts = [f"{prompt}\n{snippet}\n" for snippet in data]
    outputs = llm.generate(prompts, sampling_params)
    generated_texts = [output.outputs[0].text for output in outputs]
    print(generated_texts[:5])
    with open("./data/MAVEN_ERE/train_annotated.txt", 'wb', encoding="utf-8") as f:
        for line in generated_texts:
            f.write(f"{line}\n")

    with open("./data/MAVEN_ERE/valid_joint.txt", encoding='utf-8') as f:
        data = f.readlines()
    prompts = [f"{prompt}\n{snippet}\n" for snippet in data]
    outputs = llm.generate(prompts, sampling_params)
    generated_texts = [output.outputs[0].text for output in outputs]
    with open("./data/MAVEN_ERE/valid_annotated.txt", 'wb', encoding="utf-8") as f:
        for line in generated_texts:
            f.write(f"{line}\n")


if __name__ == "__main__":
    creating_csv()
    annotate()