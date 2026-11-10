# Benchmarks

The results of every TermXtract method on standard term extraction datasets, with the
published results of the same methods where there are any. The point is both to give
baselines for new methods and to check that the implementations behave like the
methods they implement.

This is a first version, covering the English datasets. The multilingual benchmark
(HTEC, RSDO5 and the Irish Wikipedia gold standard) will follow once those datasets
are added.

To reproduce the results, install spaCy and its `en_core_web_sm` model, and run
`python benchmarks/english.py` (see the script for the options). The scores below are
from `benchmarks/english.json`.

## English

### Setup

| Dataset | Test documents | Gold terms | Training data |
|---|---:|---:|---|
| [ACTER 1.5](https://github.com/AylaRT/ACTER), English, terms and named entities | 190 (heart failure) | 2,555 | corruption, dressage, wind energy |
| [ACL RD-TEC 2.0](http://hdl.handle.net/11372/LRT-1661), annotator 1 where both annotated | 299 | 3,027 | none |
| [GENIA term corpus 3.02](http://www.geniaproject.org/genia-corpus/term-corpus), outermost terms | 2,000 | 31,572 | none |

The settings follow ATR4S (Astrakhantsev, 2018), whose implementations of most of the
same methods are the source of the published results below. For the unsupervised
methods, the candidates are those of ATR4S's candidate collector:

- **Tags and lemmas.** The ACL RD-TEC comes with Penn Treebank tags and lemmas. ACTER,
  GENIA and the reference corpus are tagged and lemmatised with spaCy 3.8's
  `en_core_web_sm`, on the datasets' own tokens. ATR4S used EmoryNLP.
- **Lemmas.** The extractors read each word as its lemma (`lemma_layer="lemma"`), so
  "cell" and "cells" are one candidate, and the gold terms are read as the lemmas of
  the words they cover (`datasets.gold_terms(corpus, lemma_layer="lemma")`).
- **Candidates** are sequences of up to four lemmas. As in ATR4S, only the occurrences
  that pass its filters are counted, and a candidate must have at least two of them
  (`min_frequency=2`). An occurrence:
  - neither starts nor ends with a stopword;
  - passes ATR4S's noise filter (`word_pattern="atr4s"`): every word is at least three
    characters long and has only letters, digits and hyphens;
  - matches ATR4S's part-of-speech pattern, `(NN(S)?|JJ|NNP|NN(S?)IN)*(NN(S)?)`, adapted
    to spaCy's tags (`pos_pattern="(<NNS?>|<JJ>|<NNPS?>)*<NNS?|NNPS?>"`): adjectives
    and nouns, ending in a noun. The prepositions are left out, since the tags of
    ATR4S's noise filter (nouns, adjectives, verbs and adverbs) rule them out.
    Candidates may also end in a proper noun, because spaCy tags many biomedical names,
    such as HIV and AP-1, as proper nouns. With ATR4S's pattern as it is, a third of
    the occurrences of GENIA's gold terms are not counted, and Residual IDF's AvP on
    GENIA falls from 0.70 to 0.64.
- **RAKE** splits the text at its own stopwords and punctuation, so it reads the words
  as written, and its candidates only need to occur twice and have a letter or digit in
  each word.
- **The reference-corpus methods** (relevance, domain pertinence, weirdness) use the
  [Brown corpus](https://www.nltk.org/nltk_data/) (about a million words) as
  general-language text. ATR4S used the Corpus of Historical American English, which
  is not freely available. With a single reference corpus, domain pertinence ranks
  candidates as weirdness does, so their results are the same.
- **The supervised methods** are trained on the training domains of ACTER and tested on
  heart failure, so they are only run on ACTER. They read the words as written, so they
  are scored against the gold terms as written. Roger's candidates must
  occur twice and have a letter or digit in each word; token classification's are not
  filtered.
- Roger's evolutionary search is seeded (`random_state=0`), and its runs are trained
  in parallel (`n_jobs=-1`).
- Everything else is at its default.

Terms are matched to the gold standard after lowercasing, against the distinct terms of
the whole test corpus, as in the TermEval 2020 shared task. A method's ranking is that
of `evaluate`: each term is ranked by its highest score in any document.

The scores are precision (P) and F1 of the first K terms of the ranking, as percentages,
and **AvP**, average precision as ATR4S reports it: the average of the precision at
each gold term among the first *G* terms of the ranking, where *G* is the number of gold
terms among the candidates. A perfect ranking scores 1, and a random one about the share
of candidates that are gold terms, shown as **Gold** in the tables: 34% on ACTER, 52% on
the ACL RD-TEC and 53% on GENIA here, and 56% on ATR4S's candidates for the ACL RD-TEC
and GENIA.

### Comparison with published results

**ACTER, heart failure.** Lang et al. (2021, Table A1) ran ATR4S on the same test set,
with ACTER's term list including named entities.

| TermXtract | ATR4S | P@100 | ATR4S P@100 | F1@2000 | ATR4S F1@2000 | |
|---|---|---:|---:|---:|---:|---|
| tfidf | TotalTFIDF | 62.0 | 31.0 | 32.3 | 27.1 | better |
| ridf | ResidualIDF | 68.0 | 54.0 | 33.1 | 27.0 | better |
| cvalue | CValue | 59.0 | 54.0 | 34.6 | 28.0 | better |
| basic | Basic | 53.0 | 48.0 | 30.1 | 25.9 | similar or better |
| combobasic | ComboBasic | 52.0 | 48.0 | 30.0 | 25.9 | similar or better |
| relevance | Relevance | 58.0 | 52.0 | 36.7 | 30.1 | better |
| weirdness | Weirdness | 69.0 | 31.0 | 37.4 | 27.6 | better |
| topicmodeling | NovelTopicModel | 44.0 | 49.0 | 31.2 | 26.1 | worse at the top, better overall |
| domainpertinence | DomainPertinence | 69.0 | 58.0 | 37.4 | 30.3 | better |
| domaincoherence | PostRankDC | 35.0 | 31.0 | 31.2 | 26.4 | better |
| token-classification | (token classifier) | 71.0 | | 51.4\* | 58.3\* | **worse**: more precise, with lower recall |

\* F1 of all the terms tagged: precision 62.1 and recall 43.8, against precision 54.9 and recall 62.2.

The token classifier of Lang et al. (Table 3) is based on XLM-RoBERTa, as TermXtract's
is by default, and trained on English ACTER: corruption and wind energy, with dressage
for validation and choosing the best checkpoint, where TermXtract trains on all three
for three epochs. Their test set is from an earlier version of ACTER, with 2,585 gold
terms. The best system of the TermEval 2020 shared task scored 46.7 (Rigouts Terryn et
al., 2020). Roger, NMF and RAKE have no published results on these datasets.

**GENIA and the ACL RD-TEC.** ATR4S (Astrakhantsev, 2018, Table 5) reports AvP.

| TermXtract | ATR4S | GENIA | ATR4S GENIA | ACL RD-TEC | ATR4S ACL RD-TEC |
|---|---|---:|---:|---:|---:|
| basic | Basic | 0.640 | 0.644 | 0.680 | 0.697 |
| combobasic | ComboBasic | 0.641 | 0.644 | 0.699 | 0.701 |
| ridf | ResidualIDF | 0.696 | 0.705 | 0.657 | 0.677 |
| cvalue | CValue | 0.714 | 0.728 | 0.752 | 0.788 |
| weirdness | Weirdness | 0.736 | 0.767 | 0.731 | 0.758 |
| relevance | Relevance | 0.685 | 0.741 | 0.721 | 0.753 |
| topicmodeling | NovelTopicModel | 0.659 | 0.714 | 0.736 | 0.796 |
| domaincoherence | PostRankDC | 0.631 | 0.666 | 0.587 | 0.647 |

Basic, ComboBasic and Residual IDF are within 0.02 of ATR4S, C-value and weirdness
within 0.04, and relevance, topic modelling and domain coherence 0.03 to 0.06 below
it. The candidates are close to ATR4S's (52–53% gold terms, against 56%) but not the
same, since the tagger differs. The reference corpus differs too, for relevance and
weirdness, and the topic model is LDA without the Novel Topic Model's background topics
(#65). ATR4S's PostRankDC is the mean of Basic and domain coherence, where TermXtract's
`domaincoherence` is domain coherence alone; ATR4S also chooses its seed terms and
context words from the bottom of their rankings rather than the top (#52).

Counting only the occurrences that pass the filters, as ATR4S does, matters most for
C-value. Counting every occurrence and filtering the results instead gives C-value an
AvP of 0.66 on GENIA and 0.70 on the ACL RD-TEC: candidates nested in many rare or
ill-formed n-grams lose little of their frequency.

### Results

Times are in seconds on a desktop machine (32 cores, with an RTX 4090 for token
classification) and include training for the supervised methods. They do not include
tagging, which is done once and cached.

#### ACTER

| Method | Candidates | Gold | AvP (ATR4S) | P@100 | F1@100 | P@500 | F1@500 | P@1000 | F1@1000 | P@2000 | F1@2000 | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tfidf | 2,687 | 34% | 0.550 | 62.0 | 4.9 | 52.2 | 17.8 | 42.7 | 24.9 | 35.9 | 32.3 | 0 |
| ridf | 2,687 | 34% | 0.577 | 68.0 | 5.4 | 50.6 | 17.2 | 40.2 | 23.4 | 36.6 | 33.1 | 0 |
| cvalue | 2,687 | 34% | 0.504 | 59.0 | 4.7 | 42.6 | 14.5 | 37.0 | 21.5 | 38.4 | 34.6 | 0 |
| basic | 2,687 | 34% | 0.446 | 53.0 | 4.2 | 38.4 | 13.1 | 34.7 | 20.2 | 33.4 | 30.1 | 0 |
| combobasic | 2,687 | 34% | 0.437 | 52.0 | 4.1 | 38.0 | 12.9 | 32.5 | 18.9 | 33.2 | 30.0 | 0 |
| domaincoherence | 2,687 | 34% | 0.394 | 35.0 | 2.8 | 39.8 | 13.6 | 37.5 | 21.8 | 34.5 | 31.2 | 0 |
| relevance | 2,687 | 34% | 0.572 | 58.0 | 4.6 | 57.0 | 19.4 | 50.2 | 29.2 | 40.6 | 36.7 | 2 |
| domainpertinence | 2,687 | 34% | 0.635 | 69.0 | 5.4 | 61.6 | 21.0 | 52.8 | 30.7 | 41.5 | 37.4 | 2 |
| weirdness | 2,687 | 34% | 0.636 | 69.0 | 5.4 | 61.6 | 21.0 | 52.8 | 30.7 | 41.4 | 37.4 | 3 |
| topicmodeling | 2,198 | 34% | 0.422 | 44.0 | 3.5 | 37.4 | 12.7 | 35.1 | 20.4 | 34.5 | 31.2 | 0 |
| nmf | 2,687 | 34% | 0.471 | 48.0 | 3.8 | 46.2 | 15.7 | 40.7 | 23.7 | 35.4 | 32.0 | 0 |
| rake | 928 | 44% | 0.495 | 45.0 | 3.4 | 42.6 | 13.9 | 44.0 | 23.4 | 44.0 | 23.4 | 0 |
| roger | 8,989 | 6% | 0.386 | 39.0 | 2.9 | 26.4 | 8.6 | 22.1 | 12.4 | 16.9 | 14.8 | 25 |
| token-classification | 1,804 | 62% | 0.663 | 71.0 | 5.3 | 66.6 | 21.8 | 64.5 | 36.3 | 62.1 | 51.4 | 379 |

#### ACL RD-TEC 2.0

| Method | Candidates | Gold | AvP (ATR4S) | P@100 | F1@100 | P@500 | F1@500 | P@1000 | F1@1000 | P@2000 | F1@2000 | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tfidf | 1,492 | 52% | 0.688 | 73.0 | 4.9 | 65.2 | 19.4 | 57.0 | 29.6 | 52.5 | 36.0 | 0 |
| ridf | 1,492 | 52% | 0.657 | 70.0 | 4.7 | 59.0 | 17.6 | 58.8 | 30.5 | 52.5 | 36.0 | 0 |
| cvalue | 1,492 | 52% | 0.752 | 76.0 | 5.1 | 74.0 | 22.0 | 64.8 | 33.6 | 52.5 | 36.0 | 0 |
| basic | 1,492 | 52% | 0.680 | 72.0 | 4.9 | 64.6 | 19.2 | 58.0 | 30.1 | 52.5 | 36.0 | 0 |
| combobasic | 1,492 | 52% | 0.699 | 73.0 | 4.9 | 65.0 | 19.4 | 61.8 | 32.0 | 52.5 | 36.0 | 0 |
| domaincoherence | 1,492 | 52% | 0.587 | 66.0 | 4.5 | 56.2 | 16.7 | 53.1 | 27.5 | 52.5 | 36.0 | 0 |
| relevance | 1,492 | 52% | 0.721 | 78.0 | 5.3 | 69.2 | 20.6 | 64.0 | 33.2 | 52.5 | 36.0 | 2 |
| domainpertinence | 1,492 | 52% | 0.731 | 80.0 | 5.4 | 69.2 | 20.6 | 62.7 | 32.5 | 52.5 | 36.0 | 2 |
| weirdness | 1,492 | 52% | 0.731 | 80.0 | 5.4 | 69.2 | 20.6 | 62.7 | 32.5 | 52.5 | 36.0 | 2 |
| topicmodeling | 1,352 | 55% | 0.736 | 78.0 | 5.3 | 69.2 | 20.6 | 61.4 | 31.8 | 55.3 | 35.5 | 0 |
| nmf | 1,492 | 52% | 0.682 | 71.0 | 4.8 | 61.8 | 18.4 | 55.6 | 28.8 | 52.5 | 36.0 | 0 |
| rake | 219 | 38% | 0.361 | 30.0 | 1.9 | 38.4 | 5.2 | 38.4 | 5.2 | 38.4 | 5.2 | 0 |

#### GENIA

| Method | Candidates | Gold | AvP (ATR4S) | P@100 | F1@100 | P@500 | F1@500 | P@1000 | F1@1000 | P@2000 | F1@2000 | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tfidf | 15,740 | 53% | 0.718 | 93.0 | 0.6 | 84.6 | 2.7 | 82.2 | 5.2 | 75.6 | 9.3 | 1 |
| ridf | 15,740 | 53% | 0.696 | 89.0 | 0.6 | 80.4 | 2.6 | 77.6 | 4.9 | 74.6 | 9.2 | 1 |
| cvalue | 15,740 | 53% | 0.714 | 77.0 | 0.5 | 76.2 | 2.5 | 76.1 | 4.8 | 74.8 | 9.2 | 1 |
| basic | 15,740 | 53% | 0.640 | 75.0 | 0.5 | 75.4 | 2.4 | 70.7 | 4.5 | 67.6 | 8.3 | 1 |
| combobasic | 15,740 | 53% | 0.641 | 71.0 | 0.5 | 74.2 | 2.4 | 72.2 | 4.6 | 66.9 | 8.2 | 1 |
| domaincoherence | 15,740 | 53% | 0.631 | 70.0 | 0.5 | 71.2 | 2.3 | 68.6 | 4.3 | 66.1 | 8.1 | 1 |
| relevance | 15,740 | 53% | 0.685 | 89.0 | 0.6 | 78.8 | 2.5 | 75.8 | 4.8 | 71.7 | 8.8 | 3 |
| domainpertinence | 15,740 | 53% | 0.736 | 90.0 | 0.6 | 84.8 | 2.7 | 81.1 | 5.1 | 77.4 | 9.5 | 3 |
| weirdness | 15,740 | 53% | 0.736 | 90.0 | 0.6 | 84.8 | 2.7 | 81.1 | 5.1 | 77.4 | 9.5 | 3 |
| topicmodeling | 11,045 | 55% | 0.659 | 69.0 | 0.4 | 70.0 | 2.3 | 69.3 | 4.4 | 66.8 | 8.2 | 4 |
| nmf | 15,740 | 53% | 0.654 | 76.0 | 0.5 | 73.8 | 2.4 | 70.3 | 4.5 | 68.5 | 8.4 | 1 |
| rake | 4,912 | 53% | 0.582 | 61.0 | 0.4 | 59.8 | 1.9 | 58.2 | 3.6 | 56.1 | 6.7 | 0 |

### Known problems

The benchmark found these problems in the implementations. Each needs its own fix; the
results above are of the code as it is.

- `SupervisedTermExtractor.extract` trains and predicts on the same corpus, so the
  benchmark calls the extractors' own methods to train on one corpus and test on
  another (#43). The supervised methods have no `lemma_layer` yet.

C-value (#62), Basic and ComboBasic (#62), Residual IDF (#63), topic modelling (#65),
domain coherence (#52) and domain pertinence (#64) were fixed after the first version
of this benchmark, the candidates were brought in line with ATR4S's (#67, #70), NMF no
longer runs out of memory on GENIA (#66), token classification no longer fills the
disk with checkpoints (#68), and Roger trains in seconds instead of hours (#60).
AdaBoost was removed (#59): it was the weakest and slowest method, a loose
simplification of Vivaldi et al. (2001) with no published results to compare with.

### References

- Astrakhantsev, N. (2018). ATR4S: toolkit with state-of-the-art automatic terms
  recognition methods in Scala. *Language Resources and Evaluation*, 52, 853–872.
  https://arxiv.org/abs/1611.07804
- Lang, C., Wachowiak, L., Heinisch, B., & Gromann, D. (2021). Transforming term
  extraction: transformer-based approaches to multilingual term extraction across
  domains. In *Findings of ACL-IJCNLP 2021*, 3607–3620.
  https://aclanthology.org/2021.findings-acl.316
- Rigouts Terryn, A., Hoste, V., Drouin, P., & Lefever, E. (2020). TermEval 2020:
  shared task on automatic term extraction using the Annotated Corpora for Term
  Extraction Research (ACTER) dataset. In *Proceedings of CompuTerm 2020*, 85–94.
  https://aclanthology.org/2020.computerm-1.12
