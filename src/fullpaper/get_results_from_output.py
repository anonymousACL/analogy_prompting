import json
import numpy as np
import re

from os import listdir


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def find_concepts(text, concepts):
    con_count = np.zeros(4, dtype=np.uint32)

    for idx, concept in enumerate(concepts):
        if concept.lower() in text.lower():
            con_count[idx] = 1

    if sum(con_count) > 1:
        return np.zeros(4, dtype=np.uint32)
    else:
        return con_count


filepath = "results/llms/llama/"
all_choices = dict()
concept_opt = {"text": ["Up", "Down", "Left", "Right"], "pseudo": ["↑", "↓", "←", "→"]}
errors = dict()
for file in [f for f in listdir(filepath) if f.startswith("generated")]:
    filename = file.split(".txt")[0].replace("generated_answers_", "")
    errors[filename] = dict()
    if "pseudo" in filename:
        concept_type = "pseudo"
    else:
        concept_type = "text"

    # Choices/Choices Percentages
    choices = dict()

    with open(filepath+file) as f:
        results = f.read().split("----------")

    for result in results[:-1]:
        word, _, response = result.strip().split("\t")

        if "analogy" in filename:
            match = re.search(r"[cC]oncept:\** .*|[cC]hoice: .*", response)
            if match:
                response = match.group(0)
        elif "zeroshot" in filepath:
            match = re.search(r"##\n*.*\n*##", response, re.M)
            if match:
                response = match.group(0)

        res = find_concepts(response, concept_opt[concept_type])
        if sum(res) == 0:
            res = find_concepts(response.splitlines()[-1], concept_opt[concept_type]) # Trying to match an answer in the last line, especially for zeroshot
            if sum(res) == 0:
                errors[filename][word] = errors[filename].get(word, 0) + 1
                #print(response)
        choices[word] = choices.get(word, np.zeros(4, dtype=np.uint32)) + res

    all_choices[filename] = choices


for key, value in errors.items():
    print(key, round((sum(value.values())/720)*100,2), len(value), value)
with open(filepath + 'choices.json', 'w') as fp:
    json.dump(all_choices, fp, indent=2, cls=NumpyEncoder)

