# Which method should I use?

This page summarises the [benchmarks](https://github.com/TeangaNLP/TermXtract/blob/main/BENCHMARKS.md), which run every method on six datasets and compare them with published results where there are any:

- **English:** [ACTER](https://github.com/TeangaNLP/TermXtract/blob/main/BENCHMARKS.md#acter), tested on heart failure and trained on three other domains, the [ACL RD-TEC 2.0](https://github.com/TeangaNLP/TermXtract/blob/main/BENCHMARKS.md#acl-rd-tec-20) and the [GENIA](https://github.com/TeangaNLP/TermXtract/blob/main/BENCHMARKS.md#genia) term corpus;
- **Hindi:** [HTEC 2.0](https://github.com/TeangaNLP/TermXtract/blob/main/BENCHMARKS.md#htec-20-hindi), on education;
- **Slovenian:** [RSDO5](https://github.com/TeangaNLP/TermXtract/blob/main/BENCHMARKS.md#rsdo5-11-slovenian), on four domains, each tested with two of the others as training data;
- **Irish:** the [Irish Wikipedia gold standard](https://github.com/TeangaNLP/TermXtract/blob/main/BENCHMARKS.md#irish-wikipedia-gold-standard).

The numbers below are from there, where the setup is described in full. The unsupervised methods rank the candidates of [ATR4S](https://github.com/ispras/atr4s) (Astrakhantsev, 2018): sequences of up to four lemmas that match a noun-phrase pattern and occur at least twice, or, on Hindi and Irish, which have no tags or lemmas, sequences of up to four words that occur at least twice. The supervised methods are trained on the training part of each dataset, so they are not run on the ACL RD-TEC and GENIA, which have none.

## In short

| Your situation | Try |
|---|---|
| Hundreds of sentences with their terms annotated, and a GPU | [Token classification](tutorials/supervised/token_classification.md) |
| A handful of annotated sentences, and a large language model | [In-context learning](tutorials/supervised/llm.md) |
| A list of terms, but no annotated text | [Features](tutorials/supervised/features.md) |
| No training data, but general-language text in the same language | [Weirdness, domain pertinence or relevance](tutorials/reference_corpus_tutorial.md) |
| No training data and nothing to compare with | [TF-IDF](tutorials/tfidf_tutorial.md), [C-value](tutorials/cvalue_tutorial.md), or [MDERank](tutorials/embedding_tutorial.md) if you can run a transformer |
| A ranking to improve, and some terms you know | [Re-ranking](tutorials/embedding_tutorial.md#re-ranking-with-related-words) |

Whatever the method, the candidates matter as much as the scores. The unsupervised methods do best with the benchmark's settings: lemmas (`lemma_layer`), a part-of-speech pattern (`pos_pattern`) and `min_frequency=2`. The first two need a Teanga corpus tagged by a tagger such as spaCy (see [Part-of-speech patterns](tutorials/tfidf_tutorial.md#part-of-speech-patterns)).

## With training data

The supervised methods are far ahead whenever there is training data. The F1 of the terms each method extracts, against the gold terms of the test documents:

| Method | ACTER | HTEC | RSDO5 | Irish |
|---|---:|---:|---:|---:|
| token-classification | 58.2 | 58.6 | 61.5–68.2 | 65.6 |
| llm | 54.5 | 29.1 | 58.4–67.8 | 67.5 |
| features | 44.0 | 11.7 | 42.8–52.4 | 37.1 |
| roger | 18.6 | 7.2 | 15.4–18.8 | 9.6 |
| best unsupervised | 35.9 | 14.1 | 36.2–43.4 | 23.1 |

The RSDO5 column gives the range over its four domains. The unsupervised methods' F1 is over all their candidates, which is as many terms as they can find, however they rank them.

- **Token classification** fine-tunes a transformer, XLM-RoBERTa by default, to tag the terms in each sentence. It has the best F1 on every dataset but Irish. It learns from the context of the terms, so it also finds the terms that occur only once and those that the candidate patterns miss. It needs the `transformers` extra and, in practice, a GPU: training took 2 to 8 minutes per dataset on an RTX 4090.
- **In-context learning** (`llm`) prompts a large language model with the five training sentences most similar to each sentence, with their terms. With the open model `gemma4:26b`, it comes within 4 points of F1 of token classification on ACTER and RSDO5, and is ahead on Irish, where there are only nine articles to train on. It does poorly on HTEC, whose annotations are sparse and idiosyncratic, with parts of words as terms, which five demonstrations don't convey. It needs a model server, such as Ollama, and took 1 to 12 minutes per dataset.
- **Features** trains a random forest over the scores of the unsupervised methods and other features of the candidates, as HAMLET (Rigouts Terryn et al., 2021) does. It is third on every dataset but HTEC, where it classifies only 50 candidates as terms. It needs only a list of terms, not annotated text, and runs on a CPU in under a minute and a half.
- **Roger** learns a ranking of multi-word candidates from association measures. It is the weakest supervised method, below the unsupervised methods on every dataset.

### With only a few training sentences

When only 5 to 30 sentences of a domain are annotated, in-context learning beats token classification on every domain of ACTER, by 4 to 12 points of F1, and gains 7 to 14 points from 5 to 30 sentences. Adding synthetic sentences, written by a large language model or by replacing terms with similar ones from Wikipedia (see [Synthetic training data](tutorials/supervised/augmentation.md)), does not reliably help token classification: on average, each way of writing them was between 1.5 points worse and 0.7 points better than token classification alone. It helped most on wind energy, where token classification alone misses most of the terms (see [few-shot](https://github.com/TeangaNLP/TermXtract/blob/main/BENCHMARKS.md#few-shot)).

## Without training data

The unsupervised methods rank the same candidates, except for the unithood methods, which rank only multi-word candidates, and RAKE, which finds its own. So they differ in how well they rank them. The table gives the average precision (AvP, as ATR4S reports it) of each method's ranking: 1 is a perfect ranking, and a random one scores the share of its candidates that are gold terms, from 5% to 64%. The methods are sorted by their mean rank over the nine test sets, in which the four domains of RSDO5 count separately; its column gives their mean AvP. The best on each dataset is in bold.

| Method | ACTER | ACL RD-TEC | GENIA | HTEC | RSDO5 | Irish | Mean rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| domainpertinence | 0.635 | 0.731 | **0.736** | 0.500 | 0.631 | 0.317 | 3.7 |
| weirdness | **0.636** | 0.731 | **0.736** | 0.500 | 0.626 | 0.315 | 3.9 |
| relevance | 0.572 | 0.721 | 0.685 | 0.276 | **0.634** | 0.302 | 6.3 |
| tscore | 0.531 | 0.731 | 0.694 | 0.452 | 0.602 | 0.283 | 6.7 |
| mderank | 0.556 | 0.741 | **0.736** | 0.381 | 0.546 | 0.389 | 6.9 |
| tfidf | 0.550 | 0.688 | 0.718 | 0.504 | 0.521 | **0.424** | 8.0 |
| llr | 0.528 | 0.676 | 0.676 | 0.390 | 0.613 | 0.280 | 8.2 |
| cvalue | 0.504 | 0.752 | 0.714 | 0.459 | 0.553 | 0.238 | 9.0 |
| ridf | 0.577 | 0.657 | 0.696 | 0.436 | 0.556 | 0.313 | 9.2 |
| nmf | 0.471 | 0.682 | 0.654 | 0.541 | 0.534 | 0.385 | 9.3 |
| ncvalue | 0.460 | 0.730 | 0.683 | 0.403 | 0.564 | 0.247 | 10.0 |
| embedrank-mmr | 0.474 | **0.776** | 0.726 | 0.205 | 0.326 | 0.330 | 13.4 |
| topicmodeling | 0.422 | 0.736 | 0.659 | 0.343 | 0.412 | 0.275 | 14.4 |
| chi2 | 0.442 | 0.588 | 0.594 | 0.467 | 0.474 | 0.249 | 14.4 |
| basic | 0.446 | 0.680 | 0.640 | 0.382 | 0.483 | 0.206 | 15.3 |
| embedrank | 0.470 | 0.757 | 0.707 | 0.256 | 0.325 | 0.233 | 15.3 |
| pmi | 0.429 | 0.577 | 0.562 | **0.583** | 0.409 | 0.273 | 16.4 |
| yake | 0.437 | 0.648 | 0.654 | 0.240 | 0.402 | 0.305 | 16.7 |
| dice | 0.435 | 0.587 | 0.583 | 0.172 | 0.478 | 0.243 | 17.1 |
| domaincoherence | 0.394 | 0.587 | 0.631 | 0.220 | 0.487 | 0.217 | 17.6 |
| combobasic | 0.437 | 0.699 | 0.641 | 0.252 | 0.378 | 0.168 | 17.9 |
| rake | 0.475 | 0.692 | 0.674 | 0.054 | 0.215 | 0.344 | 17.9 |
| textrank | 0.339 | 0.681 | 0.584 | 0.143 | 0.253 | 0.233 | 22.1 |
| singlerank | 0.333 | 0.675 | 0.585 | 0.168 | 0.252 | 0.152 | 22.6 |
| positionrank | 0.340 | 0.671 | 0.584 | 0.156 | 0.245 | 0.207 | 22.6 |

- **The reference-corpus methods are the best choice** when you have general-language text in the same language. Weirdness, domain pertinence or relevance is the best method on five of the nine test sets, and among the six best on the others. With a single reference corpus, weirdness and domain pertinence rank alike. The benchmark compares with the Brown corpus for English, Wikipedia for Hindi and Irish, and the other domains for Slovenian. On Hindi and Irish they rank no better than TF-IDF.
- **TF-IDF and C-value are strong, fast baselines** that need nothing else. TF-IDF is the best method on Irish, and C-value the best on the ACL RD-TEC of the methods that need neither a reference corpus nor an encoder. Residual IDF and NC-value come close to them. NC-value is a little ahead of C-value on Slovenian and Irish, and behind it on the others.
- **MDERank** is among the best methods on GENIA, the ACL RD-TEC and Irish, with no reference corpus, but it needs the `transformers` extra and is slow (see [Speed](#speed)). Re-ranked by Maximal Marginal Relevance, EmbedRank is the best method on the ACL RD-TEC, but it does poorly on Hindi and Slovenian, even with a multilingual encoder.
- **The t-score and the log-likelihood ratio** rank multi-word candidates by how strongly their words stick together. They are among the five best methods on every Slovenian domain, but they never find single-word terms. PMI, Dice and χ² do worse, except PMI, which is the best method on Hindi.
- **The topic models vary** from dataset to dataset: NMF is the best method on chemistry and second on Hindi, but 14th on the ACL RD-TEC and 15th on GENIA.
- **The keyphrase methods** (RAKE, YAKE! and the three variants of TextRank) were designed to find the few key phrases of a single document, not the terms of a domain. The TextRank variants are among the six worst methods on every dataset but the ACL RD-TEC. RAKE favours long phrases: it is fourth on Irish and in the middle on English, but last on Hindi and on three of the Slovenian domains.

### Re-ranking

AdaText and SemRe-Rank revise any method's ranking by how related its terms' words are to some seed terms. They were tested on the ACL RD-TEC and GENIA, with the gold terms among the 200 most frequent candidates as seeds. AdaText improved 11 of 12 rankings, by up to 4 points of mean precision, and SemRe-Rank 10 of 12, by up to 3: far less than the papers reported. With all of the 200 most frequent candidates as seeds, without checking which are terms, they help less and sometimes hurt (see [re-ranking](https://github.com/TeangaNLP/TermXtract/blob/main/BENCHMARKS.md#re-ranking)).

## Speed

On a desktop machine, every unsupervised method but EmbedRank and MDERank takes at most a few seconds on each dataset, the largest of which, GENIA, has 2,000 abstracts. With a GPU, EmbedRank takes up to 11 seconds and MDERank up to 73, and both take much longer on a CPU. Of the supervised methods, the features method and Roger take a few seconds to two and a half minutes, and token classification and in-context learning a few minutes, on a GPU or with a model server.
