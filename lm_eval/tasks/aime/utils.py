import os
import requests
from typing import List, Dict
import re
import time
from collections import Counter
import datasets


ANSWER_PATTERN = r"(?i)Answer\s*:\s*(.*)"


EXTRACTION_TEMPLATE_IDX = r"""
Look at the following attempt by a student and extract the student's answer. If it is equivalent (ignoring trivial simplifications) to any of the provided options, return the index of that option starting from 1. Else, return -1.

Examples:

    Options: ['2x+4', '2x', '4x']
    Attempt: The answer is 3+2x.

-1
(the student's answer is not among the options)

    Options: ['72,000']
    Attempt: 72000 \text{ cents}.

1
(always give benefit of the doubt to units and ignore formatting which makes the 1st option match)

    Options: ['2/(-3)', '2/3']
    Attempt: -1 * 2/3

1
(the 1st option matches after trivial simplifications which are fine)

    Options: ['x=5']
    Attempt: 5

1

    Options: ['\dfrac{33}{100}']
    Attempt: 0.33

1

    Options: ['75^\circ']
    Attempt: ...various calculations and explanations...hence the answer is $\boxed{x in 75}$.

1

    Options: ['(1,-3)', '(1,-1)', '(1,0)', '(1,-2)']
    Attempt: -2, 1

4
(ignore whitespace and other formatting which makes the 4th option match)

    Options: ['-2,1']
    Attempt: 1, -2

1
(likely a problem where multiple solutions are possible thus ignore order)

    Options: ['11', '100', '50', '-5', '12', '10']
    Attempt: ...$\boxed{12^{\mathrm{th}}}$.

5

    Options: ['2516_8']
    Attempt: 2516

1
(give benefit of the doubt for different bases)

    Options: ['11\sqrt2']
    Attempt: 11\sqrt{2}

1

    Options: ['11,\! 111,\! 111,\! 100']
    Attempt: 11111111100

1

    Options: ['\text{Navin}']
    Attempt: ...it is navin.

1

---

YOUR TASK


Respond with only the index of the matching option starting from 1 or -1 if there is absolutely no reasonable match. Do not include a rationale.

    Options: %(expression1)s
    Attempt: %(expression2)s
""".strip()


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        if isinstance(doc["answer"], str) and doc["answer"].isdigit():
            answer = str(int(doc["answer"])) # 023 -> 23
        else:
            answer = str(doc["answer"])

        out_doc = {
            "problem": doc["problem"],
            "solution": doc["solution"],
            "answer": answer,
        }
        return out_doc

    return dataset.map(_process_doc)


class ChatCompletionSampler:
    """
    Sample from OpenRouter's chat completion API
    """

    _endpoint: str = "https://openrouter.ai/api/v1/chat/completions"
    _api_key: str = os.getenv("OPENROUTER_API_KEY", None)

    def __init__(
        self,
        model: str = "meta-llama/llama-3.3-70b-instruct",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024
    ):
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _pack_message(self, role: str, content):
        return {"role": str(role), "content": content}
    
    def __call__(self, message_list) -> str:
        if self._api_key is None:
            raise ValueError("Missing OpenRouter API key. Set OPENROUTER_API_KEY environment variable.")

        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0
        while True:
            payload = {
                "model": self.model, 
                "messages": message_list, 
                "temperature": self.temperature, 
                "max_tokens": self.max_tokens
            }
            response = requests.post(self._endpoint, headers={"Authorization": f"Bearer {self._api_key}"}, json=payload)
            if response.ok:
                return response.json()["choices"][0]["message"]["content"]
            else:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    response.status_code,
                )
                time.sleep(exception_backoff)
                trial += 1


# https://github.com/openai/simple-evals/blob/580d359553a88584c11ce4efb97d49d9386e0d9e/common.py#L153C1-L156C45
def extract_answer_idx(sampler: ChatCompletionSampler, options: List[str], attempt: str):
    prompt = EXTRACTION_TEMPLATE_IDX % {"expression1": options, "expression2": attempt}
    response = sampler([dict(content=prompt, role="user")])
    return response


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    metrics = {"exact_match": None, "extracted_answers": []}
    if isinstance(results[0], list):
        results = results[0]
        n_res = len(results) # e.g. 64
        n_res_list = [2**i for i in range(1, int(n_res.bit_length()))] # e.g. [2, 4, 8, 16, 32, 64]
        metrics = {
            **metrics,
            "exact_matches": [],
            **{f"cov@{n}": -1 for n in n_res_list},
            **{f"maj@{n}": -1 for n in n_res_list},
        }
    
    processor = os.getenv("PROCESSOR", None)
    if processor is not None:
        sampler = ChatCompletionSampler(model=os.getenv("PROCESSOR", "meta-llama/llama-3.3-70b-instruct"))
    else:
        print("Warning: no processor specified for MATH-500 evaluation.")
        sampler = None

    if isinstance(doc["answer"], str) and doc["answer"].isdigit():
        gt = str(int(doc["answer"])) # 023 -> 23
    else:
        gt = str(doc["answer"])
    
    for i, a in enumerate(results, start=1):
        if (box := last_boxed_only_string(a)) is not None:
            a = remove_boxed(box)
        elif (matches := re.findall(ANSWER_PATTERN, a, re.DOTALL)) != []:
            a = matches[-1]  # Get the last match
        else:
            print("Warning: Couldn't parse the answer; setting = 0")
            a = '0'

        if (a.isdigit()) and (gt.isdigit()):
            a = str(int(a))  # 023 -> 23
        elif sampler is not None:  # use sampler
            options = [gt] + list(set(metrics["extracted_answers"]) - {gt})
            if len(options) > 7:
                # Could switch back to exact returning like in AIME in that case
                # Problem with exact returning is that it sometimes messes up small things like a dollar sign
                print("Warning: Lots of options which may harm indexing performance:", options)
            # This ensures that if doc['answer'] is \text{Evelyn} it is represented as such and not \\text{Evelyn}
            options_str = "[" + ", ".join(["'" + str(o) + "'" for o in options]) + "]"
            idx = extract_answer_idx(sampler, options_str, a)
            if idx != "-1":
                if idx.isdigit():
                    idx = int(idx) - 1
                    if len(options) > idx >= 0:
                        a = options[idx]
                    else:
                        print("Warning: Index out of bounds; leaving answer unchanged\n", a, "\noptions", options_str, "\ndoc['answer']", gt, "\nidx", idx)
                else:
                    print("Warning: Processing did not produce integer index\na", a, "\noptions", options_str, "\ndoc['answer']", gt)
        else:
            pass

        metrics["extracted_answers"].append(a)
        a = int(a == gt)
        if not(a): # Optional logging
            print("Marked incorrect\na " + metrics["extracted_answers"][-1] + "\ndoc['answer'] " + gt)
        if i == 1:
            metrics["exact_match"] = a
            if "exact_matches" in metrics:
                metrics["exact_matches"].append(a)
        elif i > 1:
            metrics["exact_matches"].append(a)
            if i in n_res_list:
                metrics[f"cov@{i}"] = int(1 in metrics["exact_matches"])
                metrics[f"maj@{i}"] = int(gt == Counter(metrics["extracted_answers"]).most_common(1)[0][0])

    return metrics


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval
