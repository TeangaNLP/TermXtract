# Benchmarks

The results of every TermXtract method on standard term extraction datasets, with the
published results of the same methods where there are any. The point is both to give
baselines for new methods and to check that the implementations behave like the
methods they implement.

This is a first version, covering the English datasets. The multilingual benchmark
(HTEC, RSDO5 and the Irish Wikipedia gold standard) will follow once those datasets
are added.

To reproduce the results, run `python benchmarks/english.py` (see the script for the
options). The scores below are from `benchmarks/english.json`.

## English

### Setup

| Dataset | Test documents | Gold terms | Training data |
|---|---:|---:|---|
| [ACTER 1.5](https://github.com/AylaRT/ACTER), English, terms and named entities | 190 (heart failure) | 2,555 | corruption, dressage, wind energy |
| [ACL RD-TEC 2.0](http://hdl.handle.net/11372/LRT-1661), annotator 1 where both annotated | 299 | 3,027 | none |
| [GENIA term corpus 3.02](http://www.geniaproject.org/genia-corpus/term-corpus), outermost terms | 2,000 | 31,572 | none |

The settings follow ATR4S (Astrakhantsev, 2018), whose implementations of most of the
same methods are the source of the published results below:

- **Candidates** have up to four words, do not start or end with a stopword, consist of
  words with at least one letter or digit, and occur at least twice in the corpus.
  Only the first condition is part of the extractors; the benchmark applies the others
  to their output, since the extractors have no such options yet.
- On the ACL RD-TEC, whose tokens have part-of-speech tags, candidates must also match
  the `noun_phrase` pattern. ACTER and GENIA have no tags, so their candidates are not
  filtered by part of speech, unlike in ATR4S.
- The reference-corpus methods (relevance, domain pertinence, weirdness) use the
  [Brown corpus](https://www.nltk.org/nltk_data/) (about a million words) as
  general-language text. ATR4S used the Corpus of Historical American English, which
  is not freely available.
- The supervised methods are trained on the training domains of ACTER and tested on
  heart failure, so they are only run on ACTER.
- Everything else is at its default.

Terms are matched to the gold standard after lowercasing, against the distinct terms of
the whole test corpus, as in the TermEval 2020 shared task. A method's ranking is that
of `evaluate`: each term is ranked by its highest score in any document.

The scores are precision (P) and F1 of the first K terms of the ranking, as percentages,
and **AvP**, average precision as ATR4S defines it: the average precision of the first
*G* terms, where *G* is the number of gold terms among the candidates, so a perfect
ranking scores 1. AvP depends heavily on the candidates: a random ranking scores roughly
the square of the share of candidates that are gold terms, which is 0.04 on ACTER, 0.22
on the ACL RD-TEC and 0.07 on GENIA here, but about 0.31 on the candidates of ATR4S.

### Comparison with published results

**ACTER, heart failure.** Lang et al. (2021, Table A1) ran ATR4S on the same test set,
with ACTER's term list including named entities.

| TermXtract | ATR4S | P@100 | ATR4S P@100 | F1@2000 | ATR4S F1@2000 | |
|---|---|---:|---:|---:|---:|---|
| tfidf | TotalTFIDF | 67.0 | 31.0 | 30.1 | 27.1 | similar or better |
| basic | Basic | 55.0 | 48.0 | 31.1 | 25.9 | similar or better |
| combobasic | ComboBasic | 56.0 | 48.0 | 30.5 | 25.9 | similar or better |
| relevance | Relevance | 54.0 | 52.0 | 30.2 | 30.1 | similar |
| weirdness | Weirdness | 58.0 | 31.0 | 33.3 | 27.6 | similar or better |
| ridf | ResidualIDF | 23.0 | 54.0 | 16.3 | 27.0 | **worse** |
| cvalue | CValue | 27.0 | 54.0 | 16.6 | 28.0 | **worse** |
| domainpertinence | DomainPertinence | 30.0 | 58.0 | 19.6 | 30.3 | **worse** |
| topicmodeling | NovelTopicModel | 5.0 | 49.0 | 13.6 | 26.1 | **worse** |
| domaincoherence | PostRankDC | 35.3 | 31.0 | 0.5 | 26.4 | **broken** (#52) |
| token-classification | (token classifier) | | | 52.7\* | 58.3\* | similar |

\* F1 of all the terms tagged: precision 61.8 and recall 45.9, against 54.9 and 62.2.

The token classifier of Lang et al. (Table 3) is based on XLM-RoBERTa, as TermXtract's
is by default, and trained on English ACTER: corruption and wind energy, with dressage
for validation and choosing the best checkpoint, where TermXtract trains on all three
for three epochs. Their test set is from an earlier version of ACTER, with 2,585 gold
terms. Since token classification tags terms rather than ranking candidates, the
benchmark does not filter its terms by frequency. The best system of the TermEval 2020
shared task scored 46.7 (Rigouts Terryn et al., 2020). AdaBoost, Roger, NMF and RAKE
have no published results on these datasets.

**GENIA and the ACL RD-TEC.** ATR4S (Astrakhantsev, 2018, Table 5) reports AvP.

| TermXtract | ATR4S | GENIA | ATR4S GENIA | ACL RD-TEC | ATR4S ACL RD-TEC |
|---|---|---:|---:|---:|---:|
| basic | Basic | 0.274 | 0.644 | 0.314 | 0.697 |
| combobasic | ComboBasic | 0.271 | 0.644 | 0.325 | 0.701 |
| cvalue | CValue | 0.082 | 0.728 | 0.319 | 0.788 |
| relevance | Relevance | 0.174 | 0.741 | 0.397 | 0.753 |
| weirdness | Weirdness | 0.236 | 0.767 | 0.403 | 0.758 |
| ridf | ResidualIDF | not run | 0.705 | 0.204 | 0.677 |
| topicmodeling | NovelTopicModel | 0.005 | 0.714 | 0.337 | 0.796 |
| domaincoherence | PostRankDC | 0.382\* | 0.666 | 0.593\* | 0.647 |

\* Of only 14 and 57 candidates, in arbitrary order: domain coherence returns a few
dozen terms, all scored 0 (#52).

None of these are reproduced, but the comparison is weak: ATR4S's candidates were
filtered by part of speech on both datasets and grouped by lemma, so 56% of them were
gold terms, against 26% (GENIA) and 47% (ACL RD-TEC) here, and AvP rises steeply with
that share. The precision of the top-ranked terms is a fairer guide: on the ACL
RD-TEC, the methods that match ATR4S on ACTER have 70–74% precision in their first 100
terms, against 36–47% for ATR4S on a fifth of the abstracts (Lang et al., 2021, Table
A2). The ATR4S results for GENIA need part-of-speech tags to reproduce.

### Results

Times are in seconds on a desktop machine (32 cores, with an RTX 4090 for token
classification) and include training for the supervised methods.

#### ACTER

| Method | Candidates | AvP (ATR4S) | P@100 | F1@100 | P@500 | F1@500 | P@1000 | F1@1000 | P@2000 | F1@2000 | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tfidf | 6,376 | 0.201 | 67.0 | 5.0 | 51.0 | 16.7 | 40.8 | 23.0 | 34.2 | 30.1 | 1 |
| ridf | 6,376 | 0.039 | 23.0 | 1.7 | 19.0 | 6.2 | 18.3 | 10.3 | 18.6 | 16.3 | 447 |
| cvalue | 6,376 | 0.039 | 27.0 | 2.0 | 19.8 | 6.5 | 20.0 | 11.3 | 18.9 | 16.6 | 4 |
| basic | 6,376 | 0.203 | 55.0 | 4.1 | 50.8 | 16.6 | 44.1 | 24.8 | 35.4 | 31.1 | 3 |
| combobasic | 6,376 | 0.206 | 56.0 | 4.2 | 50.2 | 16.4 | 43.7 | 24.6 | 34.7 | 30.5 | 6 |
| domaincoherence | 17 | 0.222 | 35.3 | 0.5 | 35.3 | 0.5 | 35.3 | 0.5 | 35.3 | 0.5 | 3 |
| relevance | 6,376 | 0.180 | 54.0 | 4.1 | 44.6 | 14.6 | 41.4 | 23.3 | 34.4 | 30.2 | 3 |
| domainpertinence | 6,376 | 0.060 | 30.0 | 2.3 | 28.2 | 9.2 | 23.2 | 13.1 | 22.3 | 19.6 | 3 |
| weirdness | 6,376 | 0.238 | 58.0 | 4.4 | 53.8 | 17.6 | 48.2 | 27.1 | 38.0 | 33.3 | 3 |
| topicmodeling | 5,406 | 0.004 | 5.0 | 0.4 | 4.8 | 1.6 | 7.6 | 4.3 | 15.4 | 13.6 | 3 |
| nmf | 6,376 | 0.158 | 58.0 | 4.4 | 44.4 | 14.5 | 35.9 | 20.2 | 29.4 | 25.8 | 18 |
| rake | 928 | 0.217 | 45.0 | 3.4 | 42.6 | 13.9 | 44.0 | 23.4 | 44.0 | 23.4 | 2 |
| adaboost | 12,283 | 0.075 | 53.0 | 4.0 | 25.4 | 8.3 | 21.2 | 11.9 | 20.1 | 17.7 | 489 |
| roger | 8,989 | 0.089 | 34.0 | 2.6 | 25.8 | 8.4 | 20.5 | 11.5 | 15.5 | 13.6 | 9758 |
| token-classification | 1,896 | 0.432 | 73.0 | 5.5 | 68.2 | 22.3 | 64.8 | 36.5 | 61.8 | 52.7 | 383 |

#### ACL RD-TEC 2.0

| Method | Candidates | AvP (ATR4S) | P@100 | F1@100 | P@500 | F1@500 | P@1000 | F1@1000 | P@2000 | F1@2000 | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tfidf | 1,835 | 0.356 | 73.0 | 4.7 | 63.2 | 17.9 | 54.3 | 27.0 | 47.0 | 35.5 | 1 |
| ridf | 1,835 | 0.204 | 56.0 | 3.6 | 45.0 | 12.8 | 44.2 | 22.0 | 47.0 | 35.5 | 100 |
| cvalue | 1,835 | 0.319 | 62.0 | 4.0 | 60.2 | 17.1 | 52.3 | 26.0 | 47.0 | 35.5 | 1 |
| basic | 1,835 | 0.314 | 72.0 | 4.6 | 59.0 | 16.7 | 53.3 | 26.5 | 47.0 | 35.5 | 1 |
| combobasic | 1,835 | 0.325 | 72.0 | 4.6 | 60.0 | 17.0 | 53.6 | 26.6 | 47.0 | 35.5 | 2 |
| domaincoherence | 57 | 0.593 | 73.7 | 2.7 | 73.7 | 2.7 | 73.7 | 2.7 | 73.7 | 2.7 | 1 |
| relevance | 1,835 | 0.397 | 70.0 | 4.5 | 64.6 | 18.3 | 57.7 | 28.7 | 47.0 | 35.5 | 3 |
| domainpertinence | 1,835 | 0.253 | 53.0 | 3.4 | 51.0 | 14.5 | 48.7 | 24.2 | 47.0 | 35.5 | 2 |
| weirdness | 1,835 | 0.403 | 74.0 | 4.7 | 65.4 | 18.5 | 59.1 | 29.4 | 47.0 | 35.5 | 2 |
| topicmodeling | 1,628 | 0.337 | 47.0 | 3.0 | 59.4 | 16.8 | 57.9 | 28.8 | 46.9 | 32.8 | 2 |
| nmf | 1,835 | 0.329 | 70.0 | 4.5 | 56.8 | 16.1 | 52.7 | 26.2 | 47.0 | 35.5 | 11 |
| rake | 123 | 0.523 | 66.0 | 4.2 | 65.0 | 5.1 | 65.0 | 5.1 | 65.0 | 5.1 | 1 |

#### GENIA

| Method | Candidates | AvP (ATR4S) | P@100 | F1@100 | P@500 | F1@500 | P@1000 | F1@1000 | P@2000 | F1@2000 | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tfidf | 44,005 | 0.269 | 88.0 | 0.6 | 81.6 | 2.5 | 74.7 | 4.6 | 68.5 | 8.2 | 38 |
| ridf | not run: RIDF takes nearly 8 hours on GENIA (see the known problems) |
| cvalue | 44,005 | 0.082 | 35.0 | 0.2 | 45.0 | 1.4 | 39.7 | 2.4 | 26.5 | 3.2 | 27 |
| basic | 44,005 | 0.274 | 76.0 | 0.5 | 78.0 | 2.4 | 75.1 | 4.6 | 68.8 | 8.2 | 26 |
| combobasic | 44,005 | 0.271 | 77.0 | 0.5 | 77.8 | 2.4 | 75.3 | 4.6 | 68.8 | 8.2 | 47 |
| domaincoherence | 14 | 0.382 | 57.1 | 0.1 | 57.1 | 0.1 | 57.1 | 0.1 | 57.1 | 0.1 | 27 |
| relevance | 44,005 | 0.174 | 73.0 | 0.5 | 61.6 | 1.9 | 57.4 | 3.5 | 50.6 | 6.0 | 6 |
| domainpertinence | 44,005 | 0.090 | 43.0 | 0.3 | 32.0 | 1.0 | 31.1 | 1.9 | 32.2 | 3.8 | 5 |
| weirdness | 44,005 | 0.236 | 82.0 | 0.5 | 69.8 | 2.2 | 66.2 | 4.1 | 59.3 | 7.1 | 5 |
| topicmodeling | 38,322 | 0.005 | 0.0 | 0.0 | 0.4 | 0.0 | 0.2 | 0.0 | 0.4 | 0.0 | 79 |
| nmf | not run: NMF runs out of memory on GENIA (see the known problems) |
| rake | 4,912 | 0.323 | 61.0 | 0.4 | 59.8 | 1.9 | 58.2 | 3.6 | 56.1 | 6.7 | 0 |

### Known problems

The benchmark found these problems in the implementations. Each needs its own fix; the
results above are of the code as it is.

- **C-value** (`cvalue`) scores each document separately, so frequencies are those of a
  single abstract rather than of the corpus. It also counts a candidate as nested in
  another when it is a substring of it, so "art" is nested in "heart", and, as in the
  original definition, gives every single-word candidate a score of 0, where ATR4S uses
  log2(|t| + 0.1) so that single words can be ranked (#62).
- **Residual IDF** (`ridf`) computes log(observed / expected) document frequency rather
  than log(expected / observed), and the Poisson expectation uses exp(-cf) rather than
  exp(-cf / D), so it favours common n-grams. It is also quadratic, counting each
  n-gram in each document with `list.count`: 7.5 minutes on ACTER, and 7 hours 40
  minutes on GENIA, where the benchmark does not run it (#63).
- **Basic** and **ComboBasic** also score each document separately, though here this
  does not seem to hurt (#62).
- **Domain pertinence** (`domainpertinence`) divides the target frequency by the larger
  of the target and reference frequencies, so every candidate that is rarer in the
  reference corpus scores 1, and most candidates tie. In ATR4S it is the ratio of the
  two frequencies, which ranks candidates exactly as weirdness does (#64).
- **Topic modelling** (`topicmodeling`) adds up the topic probabilities of a
  candidate's words, so longer candidates score higher: its first terms on ACTER are
  "risk of heart failure" and "risk of heart" (#65).
- **Domain coherence** (`domaincoherence`) returns only a few dozen terms for a whole
  corpus (71 on the ACL RD-TEC), all scored 0 (#52).
- **NMF** (`nmf`) and domain coherence list every term of the corpus in every
  document's results. On GENIA, that is about 900,000 terms in each of 2,000 documents,
  and NMF runs out of memory, so it is not run (#66).
- **Punctuation**: candidates may contain tokens with no letters or digits, such as
  "95 % ci ,", which top the C-value ranking on ACTER, and there is no minimum
  frequency. The benchmark filters candidates itself (#67).
- **AdaBoost** (`adaboost`) is slow, mostly because of a quadratic collocation score
  (#59). **Roger** (`roger`) is slow because its evolutionary search scores every
  training candidate for every offspring: it took 2.7 hours to train on ACTER (#60).
- **Token classification** saves a checkpoint of the model, with its optimizer state,
  every 500 training steps in `./results`, several gigabytes in all for the default
  model, which filled the temporary directory of the first run (#68).
- `SupervisedTermExtractor.extract` trains and predicts on the same corpus, so the
  benchmark calls the extractors' own methods to train on one corpus and test on
  another (#43).

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
