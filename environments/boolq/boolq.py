import verifiers as vf
from datasets import load_dataset


def load_environment(
    dataset_name: str = "google/boolq",
    dataset_subset: str = "default",
    dataset_split: str = "train",
    system_prompt: str
    | None = 'Read the passage and then answer if the following question is "true" or "false" based on the information.',
) -> vf.Environment:
    dataset = load_dataset(dataset_name, dataset_subset, split=dataset_split).map(
        lambda x: {"question": x["passage"] + "\n\n" + x["question"], "info": {}}
    )

    parser = vf.MaybeThinkParser()

    def correct_answer(parser, completion, answer):
        """Binary reward for correct/incorrect answer."""
        response = parser.parse_answer(completion) or ""
        return 1.0 if response.strip() == answer.strip() else 0.0

    rubric = vf.Rubric(funcs=[correct_answer])

    return vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )
