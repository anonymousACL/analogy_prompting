import json
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from helpers import load_richardson_data

HOR_VER = True
RICHARD = False
MODEL_TYPE = "zeroshot"


def hor_ver(old_v):
    new_v = dict()
    for key, value in old_v.items():
        new_v[key] = [value[0] + value[1], value[2] + value[3]]
    return new_v


if __name__ == "__main__":
    # pseudo    ↑ ↓ ← →
    # text      up down left right
    source = f"../../data/{'richardson' if RICHARD else 'human'}_actions.txt"
    _, _, richardson_normed = load_richardson_data(source)
    concepts = list(richardson_normed.keys())

    all_results = {"choices": dict(), "choices_normed": dict(), "similarities": dict(), "spearman": dict(), "f1": dict()}
    all_choices_normed = dict()
    misses = 0
    filepath = f"results/llms/{MODEL_TYPE}/"

    with open(filepath+"choices.json") as f:
        all_results['choices'] = json.load(f)

    if HOR_VER:
        richardson_normed = hor_ver(richardson_normed)
        for filename, values in all_results['choices'].items():
            all_results['choices'][filename] = hor_ver(values)

    for file, choices in all_results['choices'].items():

        # Choices normed
        choices_normed = dict()
        for action_word, values in choices.items():
            values[values.index(max(values))] += 24 - sum(values)
            choices_normed[action_word] = [x/24 for x in values]

        all_results["choices_normed"][file] = choices_normed

        # Cosine Similarity
        similarities = dict()
        for action_word, values in choices_normed.items():
            similarities[action_word] = cosine_similarity([richardson_normed[action_word]], [values])[0][0]
        all_results["similarities"][file] = similarities

        # Spearmans Coefficient
        spearman = dict()
        richardson_concept = np.array([richardson_normed[concept] for concept in concepts]).T
        choices_concept = np.array([choices_normed[concept] for concept in concepts]).T
        for i in range(2 if HOR_VER else 4):
            spearman[i] = stats.spearmanr(richardson_concept[i], choices_concept[i])
        all_results["spearman"][file] = spearman


        # F1-Score
        richadson_choices = [richardson_normed[concept].index(max(richardson_normed[concept])) for concept in concepts]
        choices_choices = [choices_normed[concept].index(max(choices_normed[concept])) for concept in concepts]
        all_results["f1"][file] = f1_score(richadson_choices, choices_choices, labels=[0, 1] if HOR_VER else [0, 1, 2, 3], average='weighted')

    for metric, res in all_results.items():
        if "choices" not in metric:
            with open(filepath+('richardson/' if RICHARD else 'ours/')+metric+('_hor_ver' if HOR_VER else '')+'.json', 'w') as fp:
                json.dump(res, fp, indent=2)
