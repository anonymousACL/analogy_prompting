from utils.paths import ROOT
import json
import torch.nn.functional as F
import torch
import random
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

def open_json(path):
    with open(path, 'r') as file:
        loaded_dict = json.load(file)
    return loaded_dict

def dict_to_json(dict_to_save, path):
    with open(path, 'w') as file:
        json.dump(dict_to_save, file)

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


def load_richardson_data():
    with open(ROOT / "data/richardson_actions.txt", "r") as d_in:
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

def load_our_data():
    with open(ROOT / "data/human_actions.txt", "r") as d_in:
        lines = [line.split() for line in d_in.readlines()]

    output = []
    for entry in lines:
        new_entry = [convert_to_float(item) for item in entry]

        if isinstance(new_entry[1], str):
            new_entry[0] = " ".join(new_entry[:2])
            del new_entry[1]
        output.append(new_entry)

    our_data = dict()
    for elem in output:
        our_data[elem[0]] = [i for i in elem[1:]]

    # Randomizing our data
    action_words = list(our_data.keys())
    random.shuffle(action_words)

    our_categorial = dict()
    for k, v in our_data.items():
        if k == 0:
            continue
        vals = [0, 0, 0, 0]
        vals[v.index(max(v))] = 1

        our_categorial[k] = vals
    our_normed = dict()

    for action, values in our_data.items():
        if action == 0:
            continue

        our_normed[action] = [round(val / sum(values), 4) for val in values]

    return our_categorial, our_data, our_normed


def find_earliest_string(strings, target_string):
    found_strings = [s for s in strings if s in target_string.lower()]
    return min(found_strings, key=target_string.find, default=None)


# Helpers

def get_label_distribution(model_outputs):
    ans_dict = {'up': 0, 'down': 0, 'left': 0, 'right': 0}
    for event in model_outputs:
        for concept in ans_dict:
            ans_dict[concept] += model_outputs[event][concept]

    for concept in ans_dict:
        ans_dict[concept] /= len(model_outputs)
    return ans_dict

def fix_label_distributions(model_outputs):
    concepts = ['up', 'down', 'left', 'right']
    for event in model_outputs:
        concepts = [k for k in model_outputs[event]]
        percentages = [model_outputs[event][concept] for concept in concepts]
        perc_array = np.array(percentages)
        if perc_array.sum() != 100:
            if (perc_array==0).sum()==4:
                print(event)
            else:
                diff = 100 - perc_array.sum()
                model_outputs[event][concepts[np.argmax(perc_array)]] += diff
    return model_outputs

def get_corr_with_human_data(richardson_data, model_outputs, events_to_exclude=[]):
    concepts = ['up', 'down', 'left', 'right']

    model_dict = {}

    for concept in concepts:

        model_values = [model_outputs[event][concept] for event in richardson_data if event not in events_to_exclude]
        richardson_values = [richardson_data[event][concepts.index(concept)] for event in richardson_data if event not in events_to_exclude]

        corr, p_value = spearmanr(model_values, richardson_values)
        model_dict[f'{concept}_corr'] = round(corr, 2)
        model_dict[f'{concept}_p_val'] = round(p_value, 4)
        print(f"{concept.capitalize()}:\tcorr={round(corr, 2)}\tp_value={round(p_value, 4)}")

def get_corr_with_human_data_binary(richardson_data, model_outputs, events_to_exclude=[]):
    concepts = ['up', 'down', 'left', 'right']

    model_values_h = [model_outputs[event]['left'] + model_outputs[event]['right'] for event in richardson_data if event not in events_to_exclude]
    richardson_values_h = [richardson_data[event][concepts.index('left')] + richardson_data[event][concepts.index('right')] for event in richardson_data if event not in events_to_exclude]

    corr_h, p_value_h = spearmanr(model_values_h, richardson_values_h)
    print(f"Horizontal:\tcorr={round(corr_h, 2)}\tp_value={round(p_value_h, 4)}")

    model_values_v = [model_outputs[event]['up'] + model_outputs[event]['down'] for event in richardson_data if event not in events_to_exclude]
    richardson_values_v = [richardson_data[event][concepts.index('up')] + richardson_data[event][concepts.index('down')] for event in richardson_data if event not in events_to_exclude]

    corr_v, p_value_v = spearmanr(model_values_v, richardson_values_v)
    print(f"Vertical:\tcorr={round(corr_v, 2)}\tp_value={round(p_value_v, 4)}")



def get_accuracy(richardson_data, model_outputs, events_to_exclude=[]):
    events = [event for event in list(richardson_data.keys()) if event not in events_to_exclude]
    rich_array = np.stack([np.array(richardson_data[event]) for event in events])
    mod_array = np.stack([np.array([model_outputs[event]['up'], model_outputs[event]['down'],
                                    model_outputs[event]['left'], model_outputs[event]['right']]) for event in events])
    human_pref = np.argmax(rich_array, axis=1)
    model_pref = np.argmax(mod_array, axis=1)
    acc = (human_pref==model_pref).sum()/human_pref.shape[0]
    print(f'The accuracy-like alignment with human data is {round(acc, 2)}')

def get_f1_score(richardson_data, model_outputs, events_to_exclude=[]):
    events = [event for event in list(richardson_data.keys()) if event not in events_to_exclude]
    rich_array = np.stack([np.array(richardson_data[event]) for event in events])
    mod_array = np.stack([np.array([model_outputs[event]['up'], model_outputs[event]['down'],
                                    model_outputs[event]['left'], model_outputs[event]['right']]) for event in events])
    human_pref = np.argmax(rich_array, axis=1)
    model_pref = np.argmax(mod_array, axis=1)
    f1 = f1_score(human_pref, model_pref, labels=[0,1,2,3], average='weighted')
    print(f'The f1-score with human data is {round(f1, 2)}')

def get_f1_score_binary(richardson_data, model_outputs, events_to_exclude=[]):
    events = [event for event in list(richardson_data.keys()) if event not in events_to_exclude]
    rich_array = np.stack([np.array(richardson_data[event]) for event in events])
    mod_array = np.stack([np.array([model_outputs[event]['up'], model_outputs[event]['down'],
                                    model_outputs[event]['left'], model_outputs[event]['right']]) for event in events])
    rich_array_binary = np.stack([rich_array[:,0] + rich_array[:,1], rich_array[:,2] + rich_array[:,3]])
    mod_array_binary = np.stack([mod_array[:,0] + mod_array[:,1], mod_array[:,2] + mod_array[:,3]])
    human_pref = np.argmax(rich_array_binary, axis=0)
    model_pref = np.argmax(mod_array_binary, axis=0)
    f1 = f1_score(human_pref, model_pref, labels=[0,1], average='weighted')
    print(f'The binary f1-score with human data is {round(f1, 2)}')