import verifiers as vf
from datasets import load_dataset
from verifiers.utils.data_utils import extract_boxed_answer


def load_environment(
    dataset_name: str = "google/boolq",
    dataset_subset: str = "default",
    dataset_split: str = "train",
    system_prompt: str
    | None = "Read the passage and then answer the following question. Please reason step by step, then provide your final answer. Format your final answer as \\boxed{true} or \\boxed{false}.",
) -> vf.Environment:
    dataset = load_dataset(dataset_name, dataset_subset, split=dataset_split).map(
        lambda x: {"question": x["passage"] + "\n\n" + x["question"], "info": {}}
    )

    parser = vf.MaybeThinkParser(extract_fn=extract_boxed_answer)

    def correct_answer(parser, completion, answer):
        """Binary reward for correct/incorrect answer."""
        response = parser.parse_answer(completion).lower() or ""
        return 1.0 if response == str(answer).lower() else 0.0

    rubric = vf.Rubric(funcs=[correct_answer], parser=parser)

    return vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )
