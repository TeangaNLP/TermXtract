# In-Context Learning

This page uses the `corpus` and `labels` of the [overview](../supervised_tutorial.md#how-supervised-extraction-works).

The `llm` method trains nothing. It sends each sentence to a large language model in a prompt with `k` training sentences and their terms, the *demonstrations*, and reads the terms from the model's answer, as in Banerjee, Chakravarthi and McCrae (2024). With the default `selection="knn"`, the demonstrations are the training sentences most similar to the sentence, by the cosine of their [embeddings](../../api/embeddings.md) (Liu et al., 2022); `selection="random"` samples them instead.

The model is reached through a client. `OpenAIClient` speaks the OpenAI API, which local servers such as [Ollama](https://ollama.com/), vLLM and llama.cpp also offer. It needs the `llm` extra, and the "knn" selection needs the `transformers` extra:

```bash
pip install 'termxtract[llm,transformers]'
```

The example below is not run when this documentation is built, as it needs a model server. Here it uses a model served by Ollama on this machine:

```python
from termxtract import SupervisedTermExtractor
from termxtract.llm import OpenAIClient

client = OpenAIClient(
    "qwen2.5:7b", base_url="http://localhost:11434/v1", cache_dir="llm-cache"
)
extractor = SupervisedTermExtractor(method="llm", client=client, k=5)
results = extractor.fit(corpus, labels).predict(["Acute heart failure is common."])
```

Each prompt looks like this, with the most similar demonstration last:

```text
Find the domain-specific terms in the following text.

Input: Patients with chronic heart failure often receive beta-blockers.
Output: chronic heart failure, heart failure, beta-blockers

Input: Acute heart failure is common.
Output:
```

Things to know about this method:

- **The model is asked at temperature 0**, so that the same prompt gets the same answer. With `cache_dir`, every answer is kept on disk, and a prompt that was answered before is not sent again.
- **Only the terms the model finds in the sentence are kept.** The answer is split at commas (or else at newlines and semicolons), and each term is looked for in the sentence, ignoring case and spacing. Terms that are not there, such as a term the model made up, are dropped, as are numbers, terms longer than `n` words (default 10), and terms that start or end with a stopword.
- **Every extracted term has a score of 1.0**, as for token classification.
- `template` picks one of the eight prompt templates of `termxtract.supervised.llm.TEMPLATES`, which pair four instructions with two kinds of markers, `Input:`/`Output:` and `Text:`/`Terminology:`. Template 5, the default, did best in the paper, by a point or two of F1. You can also pass your own `(instruction, input marker, output marker)` tuple.
- **A sentence is never its own demonstration**, so `extract` can find the terms of the training corpus without being shown them.
- `k=0` asks the model with no demonstrations at all.
- `n_jobs` sends several prompts at once. A local server only answers them in parallel if it is set up to, for example with `OLLAMA_NUM_PARALLEL` for Ollama.

In the paper, with GPT-3.5-Turbo on ACTER, in-context learning beat fine-tuning XLM-RoBERTa by far with 5 training sentences, and by less as the training set grew: by 30 sentences the two were close. More demonstrations helped in-context learning only a little, about 2 points of F1 from 5 to 30 sentences, and kNN selection beat random selection by 1 to 2 points. With TermXtract and the open model `gemma4:26b`, in-context learning beat fine-tuning at every size up to 30 sentences on all four domains of ACTER, and with all of ACTER's training data it came within 4 points of F1 of token classification (see the [benchmarks](https://github.com/TeangaNLP/TermXtract/blob/main/BENCHMARKS.md)). So in-context learning is the method to try when only a handful of sentences are labelled, and fine-tuning when there are many.


See the [In-Context Learning API reference](../../api/supervised/llm.md) and [LLM Clients](../../api/llm.md) for all parameters.
