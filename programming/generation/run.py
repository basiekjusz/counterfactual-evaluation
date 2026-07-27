# Reference: https://github.com/allenai/open-instruct/blob/13404e37ac553fc1e072f3ac1ec12d7a5a719d13/eval/codex_humaneval/run_eval.py

import json
import os
from pathlib import Path
import random
import sys
import traceback

random.seed(0)
sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))

from programming.generation.human_eval.data import write_jsonl
from programming.generation.human_eval.evaluation import evaluate_functional_correctness
from programming.utils import one_based_indexing_unit_tests, remove_type_hints, eval_program_with_calls
from query_utils import query_batch


def load_data(data_file):
    return [json.loads(line.strip()) for line in open(data_file)]


def sanitize_completion(text):
    """BTD: extract runnable code from a chat completion.

    Chat models wrap the whole function (imports + signature + body) in a ```python fence and prose.
    Return the last fenced block's contents (language tag stripped); if there is no fence, return the
    text unchanged. Paired with a blank prompt prefix so the block runs as a standalone program.
    """
    import re
    blocks = re.findall(r"```(?:python|thonpy)?\s*\n(.*?)```", text, flags=re.DOTALL)
    return blocks[-1].strip("\n") if blocks else text


def templatize_0based(obj):
    orig_prompt = obj["prompt"]
    prompt = f"""You are an expert programmer. Complete the following function in Python 3.7. Please only output the code for the completed function.

{orig_prompt}"""
    return prompt


def templatize_1based(obj):
    orig_prompt = obj["prompt"]
    # Design decision: list[:3] returns two elements, and so does range(3). This might be slightly
    # counterintuitive, but is not uncommon in other programming languages. E.g. in Ruby,
    # `puts [1,2,3,4,5][..4]` prints five elements.
    prompt = f"""You are an expert programmer who can readily adapt to new programming languages. \
There is a new programming language, ThonPy, which is identical to Python 3.7 except all variables \
of the `list`, `tuple`, and `str` types use 1-based indexing, like in the MATLAB and R languages, where sequence indices start from 1. \
That is, index `n` represents the `n`-th element in a sequence, NOT the `n+1`-th as in 0-based indexing. \
This change only affects when the index is non-negative. \
When the index is negative, the behavior is the same as Python 3.7. \
This also affects methods of these classes such as `index` and `pop`. \
The built-in functions `enumerate` and `range` also use 1-based indexing: by default, the index of \
`enumerate` starts from 1, and so does the lower bound of `range` when not supplied (the higher bound is unchanged).

For example,
```thonpy
{one_based_indexing_unit_tests()}
```

Complete the following function in ThonPy. Please only output the code for the completed function.

{orig_prompt}"""
    return prompt


def escape(str):
    assert "\t" not in str and "\r" not in str and "\\\\n" not in str
    return str.replace("\\n", "\\\\n").replace("\n", "\\n")


def parse_bool(flag):
    if isinstance(flag, bool):
        return flag
    assert flag in {"True", "False"}
    return flag == "True"


def main(
    data_file, model_name, output_file, index_from, temperature, num_samples, k, subset_only=True, shard_idx=None
):
    index_from = int(index_from)
    temperature = float(temperature)
    num_samples = int(num_samples)
    k = int(k)
    subset_only = parse_bool(subset_only)
    data = load_data(data_file)
    if shard_idx is not None:
        data = data[: int(shard_idx) * 12]

    if subset_only:
        print(f"original # instances {len(data)}")
        filtered_data = []
        for obj in data:
            program = remove_type_hints(
                obj["prompt"]
                + obj["canonical_solution"]
                + obj["test"]
                + f'\n\ncheck({obj["entry_point"]})'
            )
            try:
                eval_program_with_calls(program, perturbation="one_based_indexing", return_output=False)
            except ZeroDivisionError:
                filtered_data.append(obj)
            except AssertionError:
                trace = traceback.format_exc()
                assert trace.strip().split("\n")[-2].endswith("in check")
                filtered_data.append(obj)
        data = filtered_data
        print(f"filtered # instances {len(data)}")

    # BTD: slice the dataset to BTD_LIMIT so the number of examples matches query_batch's truncated
    # prompt/response count; otherwise the zip(..., strict=True) below raises (responses too short).
    _btd_limit = os.environ.get("BTD_LIMIT")
    if _btd_limit is not None:
        data = data[: int(_btd_limit)]
        print(f"BTD-limited # instances {len(data)}")

    assert not os.path.exists(output_file)

    prompts = []
    for ex in data:
        if index_from == 0:
            prompt = templatize_0based(ex)
        elif index_from == 1:
            prompt = templatize_1based(ex)
        else:
            assert False
        prompts.append(prompt)

    responses = query_batch(prompts, model_name, temperature=temperature, n=num_samples)
    # BTD: with n==1 query_batch returns a flat list of strings (not list-per-prompt); wrap so the
    # flatten below doesn't iterate the string characters. (OpenRouter ignores n>1 anyway, so the
    # smoke uses num_samples=1 — see registry note.)
    if num_samples == 1:
        responses = [[r] for r in responses]
    responses = [r for ex_responses in responses for r in ex_responses]
    # BTD: chat models return a full, markdown-fenced re-definition of the function (with imports),
    # not a bare continuation. Upstream execs `prompt + completion`, which then double-defines the
    # signature -> IndentationError -> artificial pass@1=0. Strip the fence and run the completion as
    # a standalone program (prompt blanked below).
    responses = [sanitize_completion(r) for r in responses]

    # duplicates test data to match the number of outputs.
    duplicate_test_data = [example for example in data for _ in range(num_samples)]
    predictions = [
        {"task_id": example["task_id"], "prompt": example["prompt"], "completion": output}
        for example, output in zip(duplicate_test_data, responses, strict=True)
    ]
    write_jsonl(output_file, predictions)

    # BTD: pair the standalone-completion sanitising above with a blank prompt prefix so the executed
    # program is exactly the model's (sanitized) code + the test harness.
    problems = {
        example["task_id"]: {**example, "prompt": ""} for example in data
    }
    pass_at_k_results = evaluate_functional_correctness(
        output_file,
        index_from,
        k=[k],
        problems=problems,
    )

    print(pass_at_k_results)


if __name__ == "__main__":
    # BTD: the human_eval sandbox spawns a multiprocessing.Process targeting a *local* function.
    # macOS defaults to the "spawn" start method (pickles the target -> AttributeError); force "fork"
    # to match upstream's Linux behaviour. OBJC_DISABLE_INITIALIZE_FORK_SAFETY avoids macOS fork aborts.
    import multiprocessing
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
    try:
        multiprocessing.set_start_method("fork")
    except RuntimeError:
        pass
    try:
        main(*sys.argv[1:])  # pylint: disable=no-value-for-parameter,too-many-function-args
    except Exception:
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
