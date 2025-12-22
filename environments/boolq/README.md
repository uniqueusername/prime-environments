# boolq

### Overview
- **Environment ID**: `boolq`
- **Short description**: Naturally-occurring yes/no question evaluation environment
- **Tags**: text, single-turn, eval

### Datasets
- **Primary dataset(s)**: `google/boolq`, triplets of passage/question/answer
- **Source links**: [google/boolq](https://huggingface.co/datasets/google/boolq)
- **Split sizes**: Uses `train` split

### Task
- **Type**: single-turn
- **Parser**: `MaybeThinkParser` (uses `ThinkParser` if thinking, otherwise uses a basic `Parser`) 
- **Rubric overview**: Binary reward based on correct or incorrect response.

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval boolq
```

Configure model and sampling:

```bash
uv run vf-eval boolq   -m gpt-4.1-mini   -n 20 -r 3 -t 1024 -T 0.7
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_split` | str | `"train"` | Dataset split to use. |
| `dataset_subset` | str | `"default"` | Dataset subset. |
| `dataset_name` | str | `"google/boolq"` | HuggingFace dataset name. |
| `system_prompt` | str | `None` | Overrides the default prompt with a custom one if supplied. |
| `chain_of_thought` | bool | `False` | If True, the model will output a step-by-step reasoning trace before stating its answer. Otherwise, it will just output its answer. |

### Metrics
Summarize key metrics your rubric emits and how they’re interpreted.

| Metric | Meaning |
| ------ | ------- |
| `reward` | 1.0 if parsed answer equals target, else 0.0. |
