# Token Classification

This page uses the `corpus` and `labels` of the [overview](../supervised_tutorial.md#how-supervised-extraction-works).

The token-classification method fine-tunes a [Hugging Face](https://huggingface.co/) transformer model to tag each token as the beginning of a term, the continuation of a term or not part of a term. Unlike Roger, it learns from the **context** in which terms appear, and can find terms that are not in your labels.

It needs the optional `transformers` dependencies (PyTorch, transformers, accelerate and sacremoses):

```bash
pip install 'termxtract[transformers]'
```

The example below is not run when this documentation is built: the first run downloads the model (about 1 GB for `xlm-roberta-base`), and fine-tuning is slow without a GPU.

```python
from termxtract import SupervisedTermExtractor

extractor = SupervisedTermExtractor(
    method="token-classification",
    model_name="xlm-roberta-base",
    epochs=3,
    batch_size=8,
)
results = extractor.extract(corpus, labels)

for doc in results.terms:
    print(doc["doc_id"], [term["term"] for term in doc["terms"]])
```

Things to know about this method:

- **Only the terms labelled `1` are used.** Every occurrence of a labelled term in the corpus is tagged as a term, and all other tokens as non-terms, so the `0` labels make no difference. Labels are matched case-insensitively. Where labelled terms overlap, as `heart failure` and `acute heart failure` do, the longest is tagged.
- **Every extracted term has a score of 1.0.** The model tags terms rather than scoring them, so `threshold` is not used.
- **Each document lists the terms the model tagged in it.**
- `n` (default 6) is the longest term, in words, that is tagged from the labels. `max_length` (default 512) is the maximum number of model tokens per sentence; longer sentences are truncated.
- `language` (default `"en"`) selects the [Moses](https://github.com/hplt-project/sacremoses) tokenizer used to split sentences into words.
- Training saves no checkpoints: the trained model is kept in memory. To keep it, call `extractor.save(directory)`, and later pass the directory as `model_name` to extract terms without training again.
- `random_state` seeds training, so that the same data gives the same model.

To choose the number of epochs, pass a validation corpus to `fit`, labelled in the same way. The model is scored after each epoch by the F1 of the terms it tags in the validation corpus, and the best epoch is kept. With `patience`, training stops when the F1 has not improved for that many epochs:

```python
extractor = SupervisedTermExtractor(
    method="token-classification", epochs=10, patience=2, random_state=0
)
extractor.fit(train_corpus, labels, validation=validation_corpus)
```

Any Hugging Face model that supports token classification can be used as `model_name`. A model pre-trained on text from your domain usually works best.


See the [Token Classification API reference](../../api/supervised/token_classification.md) for all parameters.
