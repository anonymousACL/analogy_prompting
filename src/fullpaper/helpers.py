import torch.nn.functional as F
import torch
import random

from pynvml import *


def logprobs_from_prompt(prompt, tokenizer, model, gpu_id=False):
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = encoded["input_ids"]
    output = model(input_ids=input_ids)
    shift_labels = input_ids[..., 1:].contiguous()
    shift_logits = output.logits[..., :-1, :].contiguous()
    log_probs = []
    log_probs.append((tokenizer.decode(input_ids[0].tolist()[0]), None))
    for idx, (label_id, logit) in enumerate(zip(shift_labels[0].tolist(), shift_logits[0])):
        logit = logit.type(torch.FloatTensor) # device_map="auto" fpr model initialization
        logprob = F.log_softmax(logit, dim=0)[label_id].item()
        log_probs.append((tokenizer.decode(label_id), float(logprob)))
    return log_probs

def convert_to_float(value):
    try:
        return float(value)
    except ValueError:
        return value


def load_richardson_data(source="../../data/richardson_actions.txt"):
    with open(source, "r") as d_in:
        lines = [line.split() for line in d_in.readlines()]

    output = []
    for entry in lines:
        new_entry = [convert_to_float(item) for item in entry]

        if isinstance(new_entry[1], str):
            new_entry[0] = " ".join(new_entry[:2])
            del new_entry[1]
        output.append(new_entry)

    richardson_data = dict()
    for elem in output:
        richardson_data[elem[0]] = [i for i in elem[1:]]

    # Randomizing Richardson's data
    action_words = list(richardson_data.keys())
    random.shuffle(action_words)

    richardson_categorial = dict()
    for k, v in richardson_data.items():
        if k == 0:
            continue
        vals = [0, 0, 0, 0]
        vals[v.index(max(v))] = 1

        richardson_categorial[k] = vals
    richardson_normed = dict()

    for action, values in richardson_data.items():
        if action == 0:
            continue

        richardson_normed[action] = [round(val / sum(values), 4) for val in values]

    return richardson_categorial, richardson_data, richardson_normed


def find_earliest_string(strings, target_string):
    found_strings = [s for s in strings if s in target_string.lower()]
    return min(found_strings, key=target_string.find, default=None)


def build_max_memory_mapping():
    nvmlInit()
    mapping = dict()
    numGPUs = 0
    for i in range(torch.cuda.device_count()):
        h = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(h)
        if info.free/info.total > 0.5:
            mapping[i] = str(round(info.total / 1073741824)) + "GB"
            numGPUs += 1
        else:
            mapping[i] = "1MB"
    nvmlShutdown()
    return mapping