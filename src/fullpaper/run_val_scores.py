import torch
import numpy as np

from helpers import load_richardson_data, logprobs_from_prompt
from huggingface_hub import login
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
arrows_text = ['up', 'down', 'left', 'right']
arrows_pseudo = ['↑', '↓', '←', '→']
LLAMA_API_KEY = '<KEY>'

def hf_complete_proba(prompt, prompt_ending, model, tokenizer, verbose):
    input_prompt = prompt + prompt_ending

    res_ends = []
    proba_list = []
    if verbose: print(input_prompt)
    logprobs = logprobs_from_prompt(input_prompt, tokenizer, model)
    res = {"tokens": [x for x, y in logprobs], "token_logprobs": [y for x, y in logprobs]}
    res_ends.append(res)
    proba_list.append(np.exp(res["token_logprobs"][-1]))
    if verbose: print(prompt_ending, res, "\n")

    average_completion = 0
    for i in range(len(proba_list)):
        average_completion += (i + 1) * proba_list[i]
    average_completion = average_completion / sum(proba_list)

    return proba_list, average_completion, res_ends


def log_prob_sum(input_samples, model, tokenizer, prompt_ending):
    '''
    compute the sum of log probabilities attributed to the 7 allowed scores over x=len(input_samples) samples

    input_samples: list of prompts
    model: the model to use, either a string of a gpt model or a loaded huggingface model object
    tokenizer: huggingface tokenizer object
    prompt_ending: the string that is appended to the prompt
    hf: whether to use huggingface or gpt
    '''
    proba_sum = 0
    avg_output_list = []    #this contains probability-averaged likert scores per prompt
    for prompt in tqdm(input_samples):
        proba_list, avg_likert, _ = hf_complete_proba(prompt=prompt, model=model, tokenizer=tokenizer,
                                                      prompt_ending=prompt_ending, verbose=False)
        proba_sum += sum(proba_list)
        avg_output_list.append(avg_likert)
    return proba_sum, avg_output_list       # return sum and the list of averaged likert scores per prompt


def find_best_ending(model, tokenizer, test_inputs, prompt_ending):
    '''
    given a list of prompt endings, find the one that maximizes the probability sum over the 7 likert scores

    model: the model to use, either a string of a gpt model or a loaded huggingface model object
    tokenizer: huggingface tokenizer object
    prompt_endings: list of prompt endings to test
    hf: whether to use huggingface or gpt
    '''

    score_sums = []

    output_lists = []  # list containing the avg. generated likert scores
    print("Testing Prompt Ending:", prompt_ending)
    proba_sum, output_list = log_prob_sum(input_samples=test_inputs, model=model, tokenizer=tokenizer,
                                          prompt_ending=prompt_ending)
    score_sums.append(proba_sum)
    output_lists.append(output_list)

    # print results
    print("Average Probability:", round(proba_sum / len(test_inputs), 3))
    print()


if __name__ == "__main__":
    # loading the original human data as vectors for each action word
    _, richardson_data, richardson_normed = load_richardson_data()
    action_words = richardson_normed.keys()

    with open(LLAMA_API_KEY, "r") as f_in:
        hf_key = f_in.readline().strip()
    login(token=hf_key)

    model_name = "meta-llama/Meta-Llama-3.1-8B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)

    prompt_endings = ["CHOICE: ",
                      "Choice: ",
                      "choice: ",
                      "SELECTION: ",
                      "Selection: ",
                      "selection: ",
                      "CONCEPT: ",
                      "Concept: ",
                      "concept: "]

    print("Number of inputs used to compute probability sum:", len(action_words))

    for prompt_ending in prompt_endings:
        test_inputs = []
        for word in list(action_words):
            test_inputs.append(f"EXAMPLE TASK: Given the concepts: 'X', '-', ')', '/'. For the concept that best "
                               f"represents the event 'stopped', what concept would you choose?\n{prompt_ending} 'X'"
                               f"\n\nTASK: Given the concepts: '{'\', '.join(arrows_pseudo)}'. For the concept that "
                               f"best represents the event '{word}', what concept would you choose?\n")

        find_best_ending(model=model, tokenizer=tokenizer, test_inputs=test_inputs, prompt_ending=prompt_ending)
