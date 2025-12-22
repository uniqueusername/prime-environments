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
Document any supported environment arguments and their meaning. Example:

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `foo` | str | `"bar"` | What this controls |
| `max_examples` | int | `-1` | Limit on dataset size (use -1 for all) |

### Metrics
Summarize key metrics your rubric emits and how they’re interpreted.

| Metric | Meaning |
| ------ | ------- |
| `reward` | 1.0 if parsed answer equals target, else 0.0 |
