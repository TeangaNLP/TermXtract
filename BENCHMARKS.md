# Benchmarks

The results of every TermXtract method on standard term extraction datasets, with the
published results of the same methods where there are any. The point is both to give
baselines for new methods and to check that the implementations behave like the
methods they implement.

It covers the English datasets and three multilingual ones: HTEC (Hindi), RSDO5
(Slovenian) and the Irish Wikipedia gold standard.

To reproduce the English results, install spaCy and its `en_core_web_sm` model, and
run `python benchmarks/english.py`; for the multilingual results, run
`python benchmarks/multilingual.py` (see the scripts for the options). The scores below
are from `benchmarks/english.json` and `benchmarks/multilingual.json`.

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
- **The unithood methods** (pmi, llr, chi2, dice, tscore) only rank multi-word
  candidates, so they have fewer candidates than the other methods and never find the
  single-word terms: on ACTER and the ACL RD-TEC, all their candidates fit in the first
  2,000 terms, so they have the same F1@2000.
- **The reference-corpus methods** (relevance, domain pertinence, weirdness) use the
  [Brown corpus](https://www.nltk.org/nltk_data/) (about a million words) as
  general-language text. ATR4S used the Corpus of Historical American English, which
  is not freely available. With a single reference corpus, domain pertinence ranks
  candidates as weirdness does, so their results are the same.
- **The supervised methods** are trained on the training domains of ACTER and tested on
  heart failure, so they are only run on ACTER: `fit` learns from the gold terms of the
  training domains, and `predict` extracts the terms of the test domain. Roger learns
  from the list of gold terms of all three training domains. Token classification
  learns from each annotated occurrence, as Lang et al. (2021) do: it trains on
  corruption and wind energy for up to 10 epochs and keeps the epoch with the best F1
  on dressage, stopping after 2 epochs without improvement (`patience=2`). They read the
  words as written and are scored against the gold terms as written, as Lang et al.
  (2021) and Tran et al. (2022) score token classification, although they could read
  lemmas too (`lemma_layer`). Roger's candidates must occur twice and have a letter or digit in each word, and by default
  it drops those that start or end with a stopword; token classification's are not
  filtered. The features method learns from the gold terms of the training domains,
  as HAMLET (Rigouts Terryn et al., 2021) does: its candidates are the sequences of up
  to four words with the tag sequence of a training term (`pos_pattern="learned"`),
  with the Brown corpus as reference corpus, and it is scored on the candidates it
  classifies as terms. AdaBoost was removed (#59): it was the weakest and slowest method, a loose
  simplification of Vivaldi et al. (2001) with no published results to compare with.
- All three are seeded (`random_state=0`), and Roger's runs and the features method's
  trees are trained in parallel (`n_jobs=-1`).
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
| token-classification | (token classifier) | 78.0 | | 58.2\* | 58.3\* | **same**: more precise, with lower recall |

\* F1 of all the terms tagged: precision 61.7 and recall 55.0, against precision 54.9 and recall 62.2.

The token classifier of Lang et al. (Table 3) is based on XLM-RoBERTa, as TermXtract's
is by default, and trained in the same way, on corruption and wind energy with dressage
for validation. Validation F1 was best after the first epoch (51.5, then 33.3 and
26.5), so TermXtract's training stopped after three. Their test set is from an earlier version of ACTER, with 2,585 gold
terms. The best system of the TermEval 2020 shared task scored 46.7 (Rigouts Terryn et
al., 2020). Roger, NMF and RAKE have no published results on these datasets.

HAMLET (Rigouts Terryn et al., 2021, Table 5, setup 1) trains a random forest on the
three other English domains and tests on heart failure, with named entities as terms. It
classifies with precision 55.6, recall 37.0 and F1 44.4; the features method, over the
candidates it classifies as terms, with precision 38.1, recall 52.0 and F1 44.0. The F1
is the same, with recall and precision the other way round. HAMLET has 177 features,
against 130 here (99 of them part-of-speech features), and tunes its forest by grid
search, where the features method's forest weights the classes and has leaves of at
least 5 candidates, the settings that classified best on the training domains when one
was left out. HAMLET also used an earlier version of ACTER.

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

NC-value (Frantzi et al., 2000) has no published results on these datasets. It re-ranks
C-value by the context words of the 200 best candidates, and ranks below C-value on all
three: AvP 0.460, 0.730 and 0.683 on ACTER, the ACL RD-TEC and GENIA, against 0.504,
0.752 and 0.714.

YAKE! (Campos et al., 2020) has no published results on these datasets either. It scores
each document on its own, from the casing, position, frequency and context of the
words, and ranks below TF-IDF and C-value on every dataset (AvP 0.437, 0.648 and 0.654).
It counts words case-insensitively, so slightly more candidates occur twice, and drops
near-duplicates within each document.

TextRank (Mihalcea & Tarau, 2004), SingleRank (Wan & Xiao, 2008) and PositionRank
(Florescu & Caragea, 2017) have no published results on these datasets. They score the
words by PageRank over one co-occurrence graph of the whole corpus, and the candidates
by the sum of their words' scores. They come close to TF-IDF on the ACL RD-TEC (AvP
0.671–0.681, against 0.688), but are well below it on ACTER (0.333–0.340) and GENIA
(0.584–0.585), and the three variants differ little. On HTEC, RSDO5 and the Irish gold
standard they rank below TF-IDF, C-value and YAKE! (AvP 0.143–0.290). On the same
words and tags, their scores match those of pke 2.0.0 (Boudin, 2016) to within 1e-8
once pke's PageRank is run to convergence.

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
| ncvalue | 2,687 | 34% | 0.460 | 47.0 | 3.7 | 42.6 | 14.5 | 39.8 | 23.2 | 37.0 | 33.4 | 0 |
| basic | 2,687 | 34% | 0.446 | 53.0 | 4.2 | 38.4 | 13.1 | 34.7 | 20.2 | 33.4 | 30.1 | 0 |
| combobasic | 2,687 | 34% | 0.437 | 52.0 | 4.1 | 38.0 | 12.9 | 32.5 | 18.9 | 33.2 | 30.0 | 0 |
| domaincoherence | 2,687 | 34% | 0.394 | 35.0 | 2.8 | 39.8 | 13.6 | 37.5 | 21.8 | 34.5 | 31.2 | 0 |
| relevance | 2,687 | 34% | 0.572 | 58.0 | 4.6 | 57.0 | 19.4 | 50.2 | 29.2 | 40.6 | 36.7 | 2 |
| domainpertinence | 2,687 | 34% | 0.635 | 69.0 | 5.4 | 61.6 | 21.0 | 52.8 | 30.7 | 41.5 | 37.4 | 2 |
| weirdness | 2,687 | 34% | 0.636 | 69.0 | 5.4 | 61.6 | 21.0 | 52.8 | 30.7 | 41.4 | 37.4 | 3 |
| topicmodeling | 2,198 | 34% | 0.422 | 44.0 | 3.5 | 37.4 | 12.7 | 35.1 | 20.4 | 34.5 | 31.2 | 0 |
| nmf | 2,687 | 34% | 0.471 | 48.0 | 3.8 | 46.2 | 15.7 | 40.7 | 23.7 | 35.4 | 32.0 | 0 |
| pmi | 1,312 | 30% | 0.429 | 37.0 | 2.9 | 37.2 | 12.7 | 33.1 | 19.3 | 30.4 | 21.3 | 0 |
| llr | 1,312 | 30% | 0.528 | 56.0 | 4.4 | 41.8 | 14.2 | 35.2 | 20.5 | 30.4 | 21.3 | 0 |
| chi2 | 1,312 | 30% | 0.442 | 46.0 | 3.6 | 39.8 | 13.6 | 34.2 | 19.9 | 30.4 | 21.3 | 0 |
| dice | 1,312 | 30% | 0.435 | 46.0 | 3.6 | 38.8 | 13.2 | 33.3 | 19.4 | 30.4 | 21.3 | 0 |
| tscore | 1,312 | 30% | 0.531 | 49.0 | 3.9 | 40.2 | 13.7 | 35.1 | 20.4 | 30.4 | 21.3 | 0 |
| yake | 2,764 | 34% | 0.437 | 43.0 | 3.4 | 43.6 | 14.9 | 42.1 | 24.5 | 37.1 | 33.5 | 0 |
| textrank | 2,687 | 34% | 0.339 | 49.0 | 3.9 | 30.0 | 10.2 | 29.8 | 17.4 | 31.7 | 28.6 | 0 |
| singlerank | 2,687 | 34% | 0.333 | 28.0 | 2.2 | 32.6 | 11.1 | 29.6 | 17.2 | 33.0 | 29.8 | 0 |
| positionrank | 2,687 | 34% | 0.340 | 29.0 | 2.3 | 33.6 | 11.4 | 29.3 | 17.1 | 32.8 | 29.5 | 0 |
| rake | 928 | 44% | 0.495 | 45.0 | 3.4 | 42.6 | 13.9 | 44.0 | 23.4 | 44.0 | 23.4 | 0 |
| roger | 3,187 | 17% | 0.453 | 46.0 | 3.5 | 33.4 | 10.9 | 26.6 | 15.0 | 20.9 | 18.4 | 25 |
| features | 3,490 | 38% | 0.665 | 76.0 | 5.7 | 67.6 | 22.1 | 60.4 | 34.0 | 48.6 | 42.7 | 20 |
| token-classification | 2,280 | 62% | 0.716 | 78.0 | 5.9 | 72.6 | 23.8 | 67.8 | 38.1 | 61.9 | 54.3 | 305 |

#### ACL RD-TEC 2.0

| Method | Candidates | Gold | AvP (ATR4S) | P@100 | F1@100 | P@500 | F1@500 | P@1000 | F1@1000 | P@2000 | F1@2000 | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tfidf | 1,492 | 52% | 0.688 | 73.0 | 4.9 | 65.2 | 19.4 | 57.0 | 29.6 | 52.5 | 36.0 | 0 |
| ridf | 1,492 | 52% | 0.657 | 70.0 | 4.7 | 59.0 | 17.6 | 58.8 | 30.5 | 52.5 | 36.0 | 0 |
| cvalue | 1,492 | 52% | 0.752 | 76.0 | 5.1 | 74.0 | 22.0 | 64.8 | 33.6 | 52.5 | 36.0 | 0 |
| ncvalue | 1,492 | 52% | 0.730 | 73.0 | 4.9 | 72.0 | 21.4 | 64.0 | 33.2 | 52.5 | 36.0 | 0 |
| basic | 1,492 | 52% | 0.680 | 72.0 | 4.9 | 64.6 | 19.2 | 58.0 | 30.1 | 52.5 | 36.0 | 0 |
| combobasic | 1,492 | 52% | 0.699 | 73.0 | 4.9 | 65.0 | 19.4 | 61.8 | 32.0 | 52.5 | 36.0 | 0 |
| domaincoherence | 1,492 | 52% | 0.587 | 66.0 | 4.5 | 56.2 | 16.7 | 53.1 | 27.5 | 52.5 | 36.0 | 0 |
| relevance | 1,492 | 52% | 0.721 | 78.0 | 5.3 | 69.2 | 20.6 | 64.0 | 33.2 | 52.5 | 36.0 | 2 |
| domainpertinence | 1,492 | 52% | 0.731 | 80.0 | 5.4 | 69.2 | 20.6 | 62.7 | 32.5 | 52.5 | 36.0 | 2 |
| weirdness | 1,492 | 52% | 0.731 | 80.0 | 5.4 | 69.2 | 20.6 | 62.7 | 32.5 | 52.5 | 36.0 | 2 |
| topicmodeling | 1,352 | 55% | 0.736 | 78.0 | 5.3 | 69.2 | 20.6 | 61.4 | 31.8 | 55.3 | 35.5 | 0 |
| nmf | 1,492 | 52% | 0.682 | 71.0 | 4.8 | 61.8 | 18.4 | 55.6 | 28.8 | 52.5 | 36.0 | 0 |
| pmi | 649 | 64% | 0.577 | 53.0 | 3.6 | 63.8 | 19.0 | 63.9 | 23.7 | 63.9 | 23.7 | 0 |
| llr | 649 | 64% | 0.676 | 68.0 | 4.6 | 66.6 | 19.8 | 63.9 | 23.7 | 63.9 | 23.7 | 0 |
| chi2 | 649 | 64% | 0.588 | 55.0 | 3.7 | 65.0 | 19.4 | 63.9 | 23.7 | 63.9 | 23.7 | 0 |
| dice | 649 | 64% | 0.587 | 51.0 | 3.4 | 65.8 | 19.6 | 63.9 | 23.7 | 63.9 | 23.7 | 0 |
| tscore | 649 | 64% | 0.731 | 78.0 | 5.3 | 67.2 | 20.0 | 63.9 | 23.7 | 63.9 | 23.7 | 0 |
| yake | 1,523 | 52% | 0.648 | 59.0 | 4.0 | 65.8 | 19.6 | 60.0 | 31.1 | 52.1 | 36.3 | 0 |
| textrank | 1,492 | 52% | 0.681 | 66.0 | 4.5 | 67.2 | 20.0 | 60.1 | 31.2 | 52.5 | 36.0 | 0 |
| singlerank | 1,492 | 52% | 0.675 | 68.0 | 4.6 | 66.6 | 19.8 | 61.3 | 31.8 | 52.5 | 36.0 | 0 |
| positionrank | 1,492 | 52% | 0.671 | 68.0 | 4.6 | 66.6 | 19.8 | 61.3 | 31.8 | 52.5 | 36.0 | 0 |
| rake | 219 | 38% | 0.361 | 30.0 | 1.9 | 38.4 | 5.2 | 38.4 | 5.2 | 38.4 | 5.2 | 0 |

#### GENIA

| Method | Candidates | Gold | AvP (ATR4S) | P@100 | F1@100 | P@500 | F1@500 | P@1000 | F1@1000 | P@2000 | F1@2000 | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tfidf | 15,740 | 53% | 0.718 | 93.0 | 0.6 | 84.6 | 2.7 | 82.2 | 5.2 | 75.6 | 9.3 | 1 |
| ridf | 15,740 | 53% | 0.696 | 89.0 | 0.6 | 80.4 | 2.6 | 77.6 | 4.9 | 74.6 | 9.2 | 1 |
| cvalue | 15,740 | 53% | 0.714 | 77.0 | 0.5 | 76.2 | 2.5 | 76.1 | 4.8 | 74.8 | 9.2 | 1 |
| ncvalue | 15,740 | 53% | 0.683 | 76.0 | 0.5 | 70.6 | 2.3 | 70.4 | 4.5 | 70.2 | 8.6 | 1 |
| basic | 15,740 | 53% | 0.640 | 75.0 | 0.5 | 75.4 | 2.4 | 70.7 | 4.5 | 67.6 | 8.3 | 1 |
| combobasic | 15,740 | 53% | 0.641 | 71.0 | 0.5 | 74.2 | 2.4 | 72.2 | 4.6 | 66.9 | 8.2 | 1 |
| domaincoherence | 15,740 | 53% | 0.631 | 70.0 | 0.5 | 71.2 | 2.3 | 68.6 | 4.3 | 66.1 | 8.1 | 1 |
| relevance | 15,740 | 53% | 0.685 | 89.0 | 0.6 | 78.8 | 2.5 | 75.8 | 4.8 | 71.7 | 8.8 | 3 |
| domainpertinence | 15,740 | 53% | 0.736 | 90.0 | 0.6 | 84.8 | 2.7 | 81.1 | 5.1 | 77.4 | 9.5 | 3 |
| weirdness | 15,740 | 53% | 0.736 | 90.0 | 0.6 | 84.8 | 2.7 | 81.1 | 5.1 | 77.4 | 9.5 | 3 |
| topicmodeling | 11,045 | 55% | 0.659 | 69.0 | 0.4 | 70.0 | 2.3 | 69.3 | 4.4 | 66.8 | 8.2 | 4 |
| nmf | 15,740 | 53% | 0.654 | 76.0 | 0.5 | 73.8 | 2.4 | 70.3 | 4.5 | 68.5 | 8.4 | 1 |
| pmi | 10,548 | 54% | 0.562 | 55.0 | 0.4 | 58.4 | 1.9 | 56.9 | 3.6 | 55.3 | 6.8 | 2 |
| llr | 10,548 | 54% | 0.676 | 72.0 | 0.5 | 77.0 | 2.5 | 72.9 | 4.6 | 68.5 | 8.4 | 2 |
| chi2 | 10,548 | 54% | 0.594 | 57.0 | 0.4 | 61.2 | 2.0 | 60.7 | 3.8 | 58.5 | 7.2 | 2 |
| dice | 10,548 | 54% | 0.583 | 56.0 | 0.4 | 62.8 | 2.0 | 61.3 | 3.9 | 58.2 | 7.1 | 2 |
| tscore | 10,548 | 54% | 0.694 | 79.0 | 0.5 | 77.2 | 2.5 | 75.3 | 4.8 | 69.5 | 8.5 | 2 |
| yake | 16,129 | 54% | 0.654 | 60.0 | 0.4 | 61.4 | 2.0 | 65.1 | 4.1 | 66.5 | 8.2 | 5 |
| textrank | 15,740 | 53% | 0.584 | 61.0 | 0.4 | 58.2 | 1.9 | 60.3 | 3.8 | 59.9 | 7.4 | 1 |
| singlerank | 15,740 | 53% | 0.585 | 60.0 | 0.4 | 57.4 | 1.8 | 60.8 | 3.8 | 60.9 | 7.5 | 1 |
| positionrank | 15,740 | 53% | 0.584 | 60.0 | 0.4 | 57.0 | 1.8 | 60.5 | 3.8 | 61.1 | 7.5 | 1 |
| rake | 4,912 | 53% | 0.582 | 61.0 | 0.4 | 59.8 | 1.9 | 58.2 | 3.6 | 56.1 | 6.7 | 0 |

## Multilingual

### HTEC 2.0 (Hindi)

#### Setup

[HTEC 2.0](https://doi.org/10.5281/zenodo.24111051) has 97 segments of 65 Hindi
Wikipedia articles on education. The annotations that start
inside a word are left out, as `datasets.load("htec")` does by default.

- **Split.** HTEC has no published split, so the benchmark splits it by article, so that
  the segments of an article are all in one part, with a fixed seed. Banerjee (2025)
  used 70% of the documents for training, 10% for validation and 20% for testing, and
  the split here comes close:

  | Part | Documents | Articles | Annotated occurrences | Gold terms |
  |---|---:|---:|---:|---:|
  | Training | 68 | 20 | 3,997 | 552 |
  | Validation | 10 | 4 | 356 | 108 |
  | Test | 19 | 9 | 764 | 207 |

  Every method is scored on the test documents.
- **Candidates.** There is no Hindi tagger or lemmatiser in the benchmark, so words are
  read as written and there is no part-of-speech pattern. Candidates are sequences of up
  to four words. They must not start or end with a word of the Hindi stopword list
  (`stopwords="hindi"`, from stopwords-iso), each word must have a letter or digit, and
  they must occur at least twice in the test documents.
- **The reference corpus** is a sample of Hindi Wikipedia from the Leipzig Corpora
  Collection: 100,000 sentences, or 1.8 million words. The one sentence it shares with
  HTEC is left out.
- **The supervised methods** train on the training documents. Roger and the features
  method learn from the list of their gold terms. HTEC has no tags, so the features
  method's candidates are every sequence of up to four words that doesn't start or end
  with a stopword, and its reference corpus is the one below. Token classification
  learns from their annotated occurrences and keeps the best epoch on the validation
  documents, as for ACTER.
- **Cut-offs.** The test documents have only 207 gold terms, so precision, recall and F1
  are reported for the top 100 and 500 terms, and over all candidates.

#### Comparison with published results

Banerjee (2025, Table 6.5) reports precision, recall and F1 on HTEC 2.0 with a split of
its own:

| Method | Thesis P | R | F1 | TermXtract P | R | F1 | |
|---|---:|---:|---:|---:|---:|---:|---|
| basic | 15 | 6 | 9 | 12.0 | 5.8 | 7.8 | similar |
| combobasic | 12 | 4 | 6 | 5.0 | 2.4 | 3.3 | worse |
| cvalue | 14 | 5 | 7 | 8.0 | 3.9 | 5.2 | worse |
| nmf | 8 | 2 | 3 | 18.0 | 8.7 | 11.7 | better |
| token-classification | 72.1 | 67.8 | 69.8 | 74.9\* | 74.6\* | 74.8\* | **better** |

TermXtract's unsupervised scores here are for the top 100 terms. The thesis doesn't say
how many terms it scores, but Basic's scores match it best near 100.

\* Token classification is scored here per occurrence: an occurrence it tags counts
only if its offsets match a gold occurrence exactly. The gold occurrences are those it
learns to tag: the annotations covering whole words, with the longest kept where they
overlap. Scored over the distinct terms it tags, as for ACTER, it gets precision 61.4,
recall 56.0 and F1 58.6. The thesis's metric is not stated, so the comparison is
approximate. Its split is not published either and may be by document. Splitting by
article, as here, keeps the segments of a test article out of training, which makes
the task harder.

#### Results

| Method | Candidates | Gold | AvP (ATR4S) | P@100 | F1@100 | P@500 | F1@500 | P | R | F1 | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tfidf | 1,414 | 7% | 0.504 | 16.0 | 10.4 | 7.8 | 11.0 | 6.5 | 44.4 | 11.4 | 0 |
| ridf | 1,414 | 7% | 0.436 | 13.0 | 8.5 | 7.2 | 10.2 | 6.5 | 44.4 | 11.4 | 0 |
| cvalue | 1,414 | 7% | 0.459 | 8.0 | 5.2 | 7.8 | 11.0 | 6.5 | 44.4 | 11.4 | 0 |
| ncvalue | 1,414 | 7% | 0.403 | 9.0 | 5.9 | 8.0 | 11.3 | 6.5 | 44.4 | 11.4 | 0 |
| basic | 1,414 | 7% | 0.382 | 12.0 | 7.8 | 6.2 | 8.8 | 6.5 | 44.4 | 11.4 | 0 |
| combobasic | 1,414 | 7% | 0.252 | 5.0 | 3.3 | 5.6 | 7.9 | 6.5 | 44.4 | 11.4 | 0 |
| domaincoherence | 1,414 | 7% | 0.220 | 13.0 | 8.5 | 8.4 | 11.9 | 6.5 | 44.4 | 11.4 | 0 |
| relevance | 1,414 | 7% | 0.276 | 14.0 | 9.1 | 6.4 | 9.1 | 6.5 | 44.4 | 11.4 | 4 |
| domainpertinence | 1,414 | 7% | 0.500 | 11.0 | 7.2 | 5.8 | 8.2 | 6.5 | 44.4 | 11.4 | 3 |
| weirdness | 1,414 | 7% | 0.500 | 11.0 | 7.2 | 5.8 | 8.2 | 6.5 | 44.4 | 11.4 | 4 |
| topicmodeling | 1,257 | 6% | 0.343 | 9.0 | 5.9 | 8.2 | 11.6 | 6.4 | 38.6 | 10.9 | 0 |
| nmf | 1,414 | 7% | 0.541 | 18.0 | 11.7 | 9.0 | 12.7 | 6.5 | 44.4 | 11.4 | 0 |
| pmi | 485 | 5% | 0.583 | 6.0 | 3.9 | 5.2 | 7.2 | 5.2 | 12.1 | 7.2 | 0 |
| llr | 485 | 5% | 0.390 | 9.0 | 5.9 | 5.2 | 7.2 | 5.2 | 12.1 | 7.2 | 0 |
| chi2 | 485 | 5% | 0.467 | 7.0 | 4.6 | 5.2 | 7.2 | 5.2 | 12.1 | 7.2 | 0 |
| dice | 485 | 5% | 0.172 | 7.0 | 4.6 | 5.2 | 7.2 | 5.2 | 12.1 | 7.2 | 0 |
| tscore | 485 | 5% | 0.452 | 11.0 | 7.2 | 5.2 | 7.2 | 5.2 | 12.1 | 7.2 | 0 |
| yake | 1,411 | 7% | 0.240 | 14.0 | 9.1 | 8.4 | 11.9 | 6.5 | 44.4 | 11.4 | 0 |
| textrank | 1,414 | 7% | 0.143 | 11.0 | 7.2 | 5.8 | 8.2 | 6.5 | 44.4 | 11.4 | 0 |
| singlerank | 1,414 | 7% | 0.168 | 12.0 | 7.8 | 6.4 | 9.1 | 6.5 | 44.4 | 11.4 | 0 |
| positionrank | 1,414 | 7% | 0.156 | 12.0 | 7.8 | 6.2 | 8.8 | 6.5 | 44.4 | 11.4 | 0 |
| rake | 63 | 29% | 0.403 | 28.6 | 13.3 | 28.6 | 13.3 | 28.6 | 8.7 | 13.3 | 0 |
| roger | 485 | 5% | 0.168 | 5.0 | 3.3 | 5.2 | 7.2 | 5.2 | 12.1 | 7.2 | 18 |
| features | 50 | 30% | 0.591 | 30.0 | 11.7 | 30.0 | 11.7 | 30.0 | 7.2 | 11.7 | 25 |
| token-classification | 189 | 61% | 0.718 | 63.0 | 41.0 | 61.4 | 58.6 | 61.4 | 56.0 | 58.6 | 121 |

#### Observations

- **Few gold terms are candidates.** Only 45% of the test gold terms are candidates
  of the unsupervised methods:
  - 86 of the 207 occur only once in the test documents;
  - 28 are not sequences of whole words. Most of these are stems of inflected words,
    such as `शिक्षण संस्थान` in `शिक्षण संस्थानों`, and a few are split differently
    by the tokens, as in `बेटी पढ़ाओ-बेटी बचाओ`.
- **The reference corpus doesn't help.** Weirdness, relevance and domain pertinence do
  no better than TF-IDF. With a single reference corpus, domain pertinence ranks as
  weirdness does.
- **Roger** does worst of all, with 552 training terms and candidates that only cover
  12% of the test gold terms.
- **The features method** ranks well (AvP 0.591, second to token classification), but
  classifies only 50 candidates as terms, so its F1 is 11.7.
- **RAKE** keeps only 63 candidates, since its phrases are long and must occur twice.

### RSDO5 1.1 (Slovenian)

#### Setup

[RSDO5 1.1](http://hdl.handle.net/11356/1470) has 12 Slovenian texts, three per domain,
with every term annotated. Both in-domain and out-domain terms count, as in Tran et al.
(2022).

- **Cross-domain folds.** Each domain is the test domain in turn. For each, the
  benchmark uses the first combination of training and validation domains in Table 3
  of Tran et al. (2022):

  | Test domain | Training domains | Validation domain | Gold terms (lemmas) | Gold terms (as written) |
  |---|---|---|---:|---:|
  | Linguistics | biomechanics, chemistry | veterinary science | 3,102 | 4,601 |
  | Veterinary science | biomechanics, chemistry | linguistics | 3,380 | 4,748 |
  | Chemistry | biomechanics, veterinary science | linguistics | 1,580 | 2,409 |
  | Biomechanics | chemistry, veterinary science | linguistics | 1,596 | 2,319 |

  The gold terms as written match Tran et al.'s Table 1 exactly.
- **Candidates.** RSDO5 comes with lemmas and Universal Dependencies tags, so the
  unsupervised methods use the candidates of the English benchmark: sequences of up to
  four lemmas that match the `noun_phrase` pattern over `upos`, pass ATR4S's noise
  filter and occur at least twice. They are scored against the lemmatised gold terms;
  RAKE and the supervised methods, which find words as written, against the gold terms
  as written.
- **Stopwords.** Candidates and extracted terms must not start or end with a word of
  stopwords-iso's Slovenian list (`stopwords="slovenian"`), which is also RAKE's
  stoplist.
- **The reference corpus** of relevance, weirdness and domain pertinence is the other
  three domains. Domain pertinence compares the test domain with each of them, as
  Navigli & Velardi (2004) do with contrasting domains; weirdness and relevance read
  them as one corpus.
- **The supervised methods** train on the two training domains. Token classification
  learns from their annotated occurrences and keeps the best epoch on the validation
  domain, with the settings used for ACTER. The features method learns the tag
  sequences of the training terms over `upos`, as on ACTER, but has no reference
  corpus, since the other domains include the training domains.

#### Comparison with published results

Tran et al. (2022, Table 3) report XLM-R token classification on the same folds, with
precision, recall and F1 over the distinct terms of the test domain, lowercased, as
TermXtract's token classification is scored here and on ACTER:

| Test domain | Tran et al. P | R | F1 | TermXtract P | R | F1 | |
|---|---:|---:|---:|---:|---:|---:|---|
| Linguistics | 69.6 | 64.1 | 66.7 | 69.0 | 66.5 | 67.7 | same |
| Veterinary science | 71.1 | 66.7 | 68.8 | 72.0 | 64.8 | 68.2 | same |
| Chemistry | 68.7 | 55.1 | 61.2 | 65.8 | 60.9 | 63.2 | better |
| Biomechanics | 63.5 | 66.8 | 65.1 | 66.7 | 57.0 | 61.5 | worse |

They train for up to 20 epochs with a batch size of 32, where TermXtract trains for up
to 10 with a batch size of 8. Tran et al. also report the earlier Slovenian system of
Ljubešić et al. (2019) at 30–38 F1 on these domains; over all their candidates, the
unsupervised methods here reach 36–43 (TF-IDF, for example: 36.2 on linguistics, 43.4
on veterinary science, 37.8 on chemistry and 40.4 on biomechanics).

#### Results

##### Linguistics

| Method | Candidates | Gold | AvP (ATR4S) | P@100 | F1@100 | P@500 | F1@500 | P@1000 | F1@1000 | P@2000 | F1@2000 | P | R | F1 | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tfidf | 4,045 | 32% | 0.427 | 54.0 | 3.4 | 40.8 | 11.3 | 36.4 | 17.7 | 33.5 | 26.3 | 32.0 | 41.7 | 36.2 | 1 |
| ridf | 4,045 | 32% | 0.530 | 69.0 | 4.3 | 51.4 | 14.3 | 42.6 | 20.8 | 36.5 | 28.6 | 32.0 | 41.7 | 36.2 | 1 |
| cvalue | 4,045 | 32% | 0.524 | 67.0 | 4.2 | 47.4 | 13.2 | 38.2 | 18.6 | 37.9 | 29.7 | 32.0 | 41.7 | 36.2 | 1 |
| ncvalue | 4,045 | 32% | 0.538 | 67.0 | 4.2 | 50.2 | 13.9 | 40.8 | 19.9 | 36.1 | 28.3 | 32.0 | 41.7 | 36.2 | 1 |
| basic | 4,045 | 32% | 0.473 | 57.0 | 3.6 | 45.2 | 12.5 | 41.3 | 20.1 | 35.0 | 27.5 | 32.0 | 41.7 | 36.2 | 1 |
| combobasic | 4,045 | 32% | 0.374 | 42.0 | 2.6 | 36.0 | 10.0 | 35.3 | 17.2 | 33.4 | 26.1 | 32.0 | 41.7 | 36.2 | 1 |
| domaincoherence | 4,045 | 32% | 0.482 | 58.0 | 3.6 | 46.8 | 13.0 | 40.4 | 19.7 | 32.9 | 25.8 | 32.0 | 41.7 | 36.2 | 1 |
| relevance | 4,045 | 32% | 0.567 | 69.0 | 4.3 | 55.2 | 15.3 | 47.7 | 23.3 | 39.2 | 30.7 | 32.0 | 41.7 | 36.2 | 1 |
| domainpertinence | 4,045 | 32% | 0.596 | 72.0 | 4.5 | 58.2 | 16.2 | 47.2 | 23.0 | 38.6 | 30.3 | 32.0 | 41.7 | 36.2 | 1 |
| weirdness | 4,045 | 32% | 0.591 | 73.0 | 4.6 | 57.4 | 15.9 | 47.6 | 23.2 | 38.6 | 30.3 | 32.0 | 41.7 | 36.2 | 1 |
| topicmodeling | 2,019 | 32% | 0.438 | 47.0 | 2.9 | 33.2 | 9.2 | 31.1 | 15.2 | 32.0 | 25.1 | 32.2 | 21.0 | 25.4 | 1 |
| nmf | 4,045 | 32% | 0.505 | 64.0 | 4.0 | 47.0 | 13.0 | 38.5 | 18.8 | 34.4 | 26.9 | 32.0 | 41.7 | 36.2 | 1 |
| pmi | 2,205 | 32% | 0.389 | 25.0 | 1.6 | 40.8 | 11.3 | 38.2 | 18.6 | 33.4 | 26.2 | 32.4 | 23.0 | 26.9 | 1 |
| llr | 2,205 | 32% | 0.547 | 65.0 | 4.1 | 48.0 | 13.3 | 42.7 | 20.8 | 33.9 | 26.6 | 32.4 | 23.0 | 26.9 | 1 |
| chi2 | 2,205 | 32% | 0.415 | 31.0 | 1.9 | 45.2 | 12.5 | 40.6 | 19.8 | 33.7 | 26.4 | 32.4 | 23.0 | 26.9 | 1 |
| dice | 2,205 | 32% | 0.413 | 32.0 | 2.0 | 43.8 | 12.2 | 41.7 | 20.3 | 34.1 | 26.7 | 32.4 | 23.0 | 26.9 | 1 |
| tscore | 2,205 | 32% | 0.574 | 64.0 | 4.0 | 48.0 | 13.3 | 38.5 | 18.8 | 33.8 | 26.5 | 32.4 | 23.0 | 26.9 | 1 |
| yake | 4,033 | 32% | 0.367 | 39.0 | 2.4 | 34.6 | 9.6 | 33.3 | 16.2 | 32.8 | 25.7 | 31.9 | 41.5 | 36.0 | 7 |
| textrank | 4,045 | 32% | 0.234 | 10.0 | 0.6 | 21.6 | 6.0 | 28.0 | 13.7 | 32.9 | 25.8 | 32.0 | 41.7 | 36.2 | 1 |
| singlerank | 4,045 | 32% | 0.240 | 15.0 | 0.9 | 23.4 | 6.5 | 27.0 | 13.2 | 32.8 | 25.7 | 32.0 | 41.7 | 36.2 | 1 |
| positionrank | 4,045 | 32% | 0.244 | 16.0 | 1.0 | 24.2 | 6.7 | 25.9 | 12.6 | 31.4 | 24.6 | 32.0 | 41.7 | 36.2 | 1 |
| rake | 1,876 | 24% | 0.285 | 33.0 | 1.4 | 23.6 | 4.6 | 23.4 | 8.4 | 23.5 | 13.6 | 23.5 | 9.6 | 13.6 | 3 |
| roger | 5,669 | 15% | 0.422 | 51.0 | 2.2 | 34.4 | 6.7 | 28.9 | 10.3 | 22.2 | 13.5 | 14.9 | 18.3 | 16.4 | 29 |
| features | 10,170 | 35% | 0.570 | 70.0 | 3.0 | 61.6 | 12.1 | 57.3 | 20.5 | 54.9 | 33.3 | 34.8 | 76.9 | 47.9 | 9 |
| token-classification | 4,431 | 69% | 0.717 | 81.0 | 3.4 | 74.2 | 14.5 | 71.6 | 25.6 | 68.2 | 41.4 | 69.0 | 66.5 | 67.7 | 491 |

##### Veterinary science

| Method | Candidates | Gold | AvP (ATR4S) | P@100 | F1@100 | P@500 | F1@500 | P@1000 | F1@1000 | P@2000 | F1@2000 | P | R | F1 | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tfidf | 3,764 | 41% | 0.584 | 69.0 | 4.0 | 58.4 | 15.1 | 53.6 | 24.5 | 44.9 | 33.3 | 41.2 | 45.9 | 43.4 | 1 |
| ridf | 3,764 | 41% | 0.632 | 76.0 | 4.4 | 64.4 | 16.6 | 54.6 | 24.9 | 48.0 | 35.7 | 41.2 | 45.9 | 43.4 | 1 |
| cvalue | 3,764 | 41% | 0.549 | 66.0 | 3.8 | 54.8 | 14.1 | 45.0 | 20.5 | 45.8 | 34.0 | 41.2 | 45.9 | 43.4 | 1 |
| ncvalue | 3,764 | 41% | 0.568 | 70.0 | 4.0 | 59.0 | 15.2 | 47.0 | 21.5 | 45.6 | 33.9 | 41.2 | 45.9 | 43.4 | 1 |
| basic | 3,764 | 41% | 0.489 | 52.0 | 3.0 | 48.8 | 12.6 | 46.4 | 21.2 | 41.6 | 31.0 | 41.2 | 45.9 | 43.4 | 1 |
| combobasic | 3,764 | 41% | 0.375 | 37.0 | 2.1 | 34.6 | 8.9 | 38.9 | 17.8 | 39.6 | 29.4 | 41.2 | 45.9 | 43.4 | 1 |
| domaincoherence | 3,764 | 41% | 0.472 | 59.0 | 3.4 | 45.8 | 11.8 | 41.0 | 18.7 | 40.3 | 30.0 | 41.2 | 45.9 | 43.4 | 1 |
| relevance | 3,764 | 41% | 0.714 | 88.0 | 5.1 | 71.8 | 18.5 | 63.4 | 28.9 | 51.3 | 38.1 | 41.2 | 45.9 | 43.4 | 2 |
| domainpertinence | 3,764 | 41% | 0.706 | 87.0 | 5.0 | 72.2 | 18.6 | 62.0 | 28.3 | 51.2 | 38.1 | 41.2 | 45.9 | 43.4 | 2 |
| weirdness | 3,764 | 41% | 0.710 | 88.0 | 5.1 | 72.6 | 18.7 | 61.6 | 28.1 | 50.9 | 37.9 | 41.2 | 45.9 | 43.4 | 1 |
| topicmodeling | 1,831 | 39% | 0.387 | 43.0 | 2.5 | 34.2 | 8.8 | 36.4 | 16.6 | 38.5 | 27.1 | 38.5 | 20.9 | 27.1 | 1 |
| nmf | 3,764 | 41% | 0.498 | 48.0 | 2.8 | 47.6 | 12.3 | 48.6 | 22.2 | 51.1 | 38.0 | 41.2 | 45.9 | 43.4 | 1 |
| pmi | 1,941 | 38% | 0.483 | 44.0 | 2.5 | 50.4 | 13.0 | 45.9 | 21.0 | 37.8 | 27.6 | 37.8 | 21.7 | 27.6 | 1 |
| llr | 1,941 | 38% | 0.666 | 79.0 | 4.5 | 57.6 | 14.8 | 48.3 | 22.1 | 37.8 | 27.6 | 37.8 | 21.7 | 27.6 | 1 |
| chi2 | 1,941 | 38% | 0.560 | 55.0 | 3.2 | 56.0 | 14.4 | 48.2 | 22.0 | 37.8 | 27.6 | 37.8 | 21.7 | 27.6 | 1 |
| dice | 1,941 | 38% | 0.564 | 58.0 | 3.3 | 57.6 | 14.8 | 50.0 | 22.8 | 37.8 | 27.6 | 37.8 | 21.7 | 27.6 | 1 |
| tscore | 1,941 | 38% | 0.634 | 69.0 | 4.0 | 56.0 | 14.4 | 45.5 | 20.8 | 37.8 | 27.6 | 37.8 | 21.7 | 27.6 | 1 |
| yake | 3,754 | 41% | 0.418 | 37.0 | 2.1 | 40.6 | 10.5 | 42.0 | 19.2 | 41.6 | 31.0 | 41.2 | 45.8 | 43.4 | 5 |
| textrank | 3,764 | 41% | 0.277 | 13.0 | 0.7 | 22.8 | 5.9 | 27.5 | 12.6 | 38.5 | 28.6 | 41.2 | 45.9 | 43.4 | 1 |
| singlerank | 3,764 | 41% | 0.284 | 17.0 | 1.0 | 24.0 | 6.2 | 28.7 | 13.1 | 39.5 | 29.3 | 41.2 | 45.9 | 43.4 | 1 |
| positionrank | 3,764 | 41% | 0.275 | 17.0 | 1.0 | 23.2 | 6.0 | 27.7 | 12.6 | 38.2 | 28.4 | 41.2 | 45.9 | 43.4 | 1 |
| rake | 998 | 35% | 0.369 | 32.0 | 1.3 | 24.2 | 4.6 | 35.4 | 12.3 | 35.4 | 12.3 | 35.4 | 7.4 | 12.3 | 1 |
| roger | 5,040 | 17% | 0.433 | 49.0 | 2.0 | 37.2 | 7.1 | 29.6 | 10.3 | 23.8 | 14.1 | 17.0 | 18.1 | 17.5 | 28 |
| features | 7,500 | 43% | 0.650 | 72.0 | 3.0 | 70.2 | 13.4 | 67.5 | 23.5 | 60.7 | 36.0 | 42.8 | 67.6 | 52.4 | 8 |
| token-classification | 4,274 | 72% | 0.709 | 88.0 | 3.6 | 69.2 | 13.2 | 70.3 | 24.5 | 68.0 | 40.3 | 72.0 | 64.8 | 68.2 | 339 |

##### Chemistry

| Method | Candidates | Gold | AvP (ATR4S) | P@100 | F1@100 | P@500 | F1@500 | P@1000 | F1@1000 | P@2000 | F1@2000 | P | R | F1 | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tfidf | 2,961 | 29% | 0.609 | 74.0 | 8.8 | 52.0 | 25.0 | 42.7 | 33.1 | 31.8 | 35.5 | 29.0 | 54.4 | 37.8 | 1 |
| ridf | 2,961 | 29% | 0.572 | 70.0 | 8.3 | 47.4 | 22.8 | 38.6 | 29.9 | 32.3 | 36.1 | 29.0 | 54.4 | 37.8 | 1 |
| cvalue | 2,961 | 29% | 0.615 | 71.0 | 8.5 | 49.2 | 23.7 | 35.3 | 27.4 | 33.2 | 37.2 | 29.0 | 54.4 | 37.8 | 1 |
| ncvalue | 2,961 | 29% | 0.618 | 71.0 | 8.5 | 51.0 | 24.5 | 38.5 | 29.8 | 32.5 | 36.3 | 29.0 | 54.4 | 37.8 | 1 |
| basic | 2,961 | 29% | 0.534 | 54.0 | 6.4 | 49.4 | 23.8 | 37.9 | 29.4 | 31.4 | 35.1 | 29.0 | 54.4 | 37.8 | 1 |
| combobasic | 2,961 | 29% | 0.420 | 50.0 | 6.0 | 35.4 | 17.0 | 32.8 | 25.4 | 30.1 | 33.7 | 29.0 | 54.4 | 37.8 | 1 |
| domaincoherence | 2,961 | 29% | 0.532 | 67.0 | 8.0 | 43.4 | 20.9 | 34.9 | 27.1 | 30.7 | 34.3 | 29.0 | 54.4 | 37.8 | 1 |
| relevance | 2,961 | 29% | 0.671 | 83.0 | 9.9 | 57.4 | 27.6 | 43.7 | 33.9 | 30.8 | 34.4 | 29.0 | 54.4 | 37.8 | 2 |
| domainpertinence | 2,961 | 29% | 0.657 | 80.0 | 9.5 | 57.8 | 27.8 | 43.5 | 33.7 | 31.6 | 35.3 | 29.0 | 54.4 | 37.8 | 2 |
| weirdness | 2,961 | 29% | 0.647 | 77.0 | 9.2 | 55.4 | 26.6 | 43.1 | 33.4 | 30.9 | 34.5 | 29.0 | 54.4 | 37.8 | 2 |
| topicmodeling | 1,775 | 31% | 0.448 | 43.0 | 5.1 | 40.0 | 19.2 | 36.6 | 28.4 | 31.4 | 33.3 | 31.4 | 35.3 | 33.3 | 1 |
| nmf | 2,961 | 29% | 0.682 | 81.0 | 9.6 | 57.6 | 27.7 | 44.9 | 34.8 | 34.8 | 38.9 | 29.0 | 54.4 | 37.8 | 1 |
| pmi | 1,771 | 27% | 0.291 | 23.0 | 2.7 | 29.2 | 14.0 | 29.1 | 22.6 | 27.3 | 28.8 | 27.3 | 30.6 | 28.8 | 1 |
| llr | 1,771 | 27% | 0.667 | 67.0 | 8.0 | 48.2 | 23.2 | 35.8 | 27.8 | 27.3 | 28.8 | 27.3 | 30.6 | 28.8 | 1 |
| chi2 | 1,771 | 27% | 0.386 | 38.0 | 4.5 | 39.2 | 18.8 | 33.4 | 25.9 | 27.3 | 28.8 | 27.3 | 30.6 | 28.8 | 1 |
| dice | 1,771 | 27% | 0.398 | 39.0 | 4.6 | 41.8 | 20.1 | 32.8 | 25.4 | 27.3 | 28.8 | 27.3 | 30.6 | 28.8 | 1 |
| tscore | 1,771 | 27% | 0.659 | 66.0 | 7.9 | 48.8 | 23.5 | 36.2 | 28.1 | 27.3 | 28.8 | 27.3 | 30.6 | 28.8 | 1 |
| yake | 2,941 | 29% | 0.450 | 46.0 | 5.5 | 42.2 | 20.3 | 37.4 | 29.0 | 31.6 | 35.3 | 28.7 | 53.5 | 37.4 | 5 |
| textrank | 2,961 | 29% | 0.211 | 7.0 | 0.8 | 23.2 | 11.2 | 26.2 | 20.3 | 29.3 | 32.8 | 29.0 | 54.4 | 37.8 | 1 |
| singlerank | 2,961 | 29% | 0.233 | 4.0 | 0.5 | 28.6 | 13.8 | 28.2 | 21.9 | 30.4 | 34.0 | 29.0 | 54.4 | 37.8 | 1 |
| positionrank | 2,961 | 29% | 0.234 | 6.0 | 0.7 | 28.2 | 13.6 | 28.7 | 22.2 | 30.5 | 34.1 | 29.0 | 54.4 | 37.8 | 1 |
| rake | 923 | 18% | 0.315 | 16.0 | 1.3 | 21.2 | 7.3 | 18.4 | 10.2 | 18.4 | 10.2 | 18.4 | 7.1 | 10.2 | 2 |
| roger | 6,189 | 11% | 0.323 | 34.0 | 2.7 | 22.6 | 7.8 | 15.3 | 9.0 | 13.8 | 12.5 | 10.7 | 27.6 | 15.4 | 31 |
| features | 6,323 | 30% | 0.527 | 66.0 | 5.3 | 53.8 | 18.5 | 46.2 | 27.1 | 41.3 | 37.5 | 29.6 | 77.6 | 42.8 | 8 |
| token-classification | 2,230 | 66% | 0.703 | 80.0 | 6.4 | 69.8 | 24.0 | 68.4 | 40.1 | 66.0 | 59.9 | 65.8 | 60.9 | 63.2 | 190 |

##### Biomechanics

| Method | Candidates | Gold | AvP (ATR4S) | P@100 | F1@100 | P@500 | F1@500 | P@1000 | F1@1000 | P@2000 | F1@2000 | P | R | F1 | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tfidf | 2,202 | 35% | 0.465 | 45.0 | 5.3 | 47.6 | 22.7 | 43.9 | 33.8 | 35.0 | 39.0 | 34.8 | 48.1 | 40.4 | 1 |
| ridf | 2,202 | 35% | 0.489 | 58.0 | 6.8 | 43.2 | 20.6 | 38.5 | 29.7 | 35.4 | 39.4 | 34.8 | 48.1 | 40.4 | 1 |
| cvalue | 2,202 | 35% | 0.526 | 56.0 | 6.6 | 50.0 | 23.9 | 42.9 | 33.1 | 37.6 | 41.9 | 34.8 | 48.1 | 40.4 | 1 |
| ncvalue | 2,202 | 35% | 0.532 | 60.0 | 7.1 | 48.6 | 23.2 | 41.4 | 31.9 | 36.2 | 40.3 | 34.8 | 48.1 | 40.4 | 1 |
| basic | 2,202 | 35% | 0.438 | 44.0 | 5.2 | 43.6 | 20.8 | 38.4 | 29.6 | 36.0 | 40.1 | 34.8 | 48.1 | 40.4 | 1 |
| combobasic | 2,202 | 35% | 0.343 | 34.0 | 4.0 | 34.2 | 16.3 | 36.5 | 28.1 | 36.0 | 40.1 | 34.8 | 48.1 | 40.4 | 1 |
| domaincoherence | 2,202 | 35% | 0.463 | 58.0 | 6.8 | 36.8 | 17.6 | 33.8 | 26.0 | 34.3 | 38.2 | 34.8 | 48.1 | 40.4 | 2 |
| relevance | 2,202 | 35% | 0.583 | 68.0 | 8.0 | 51.6 | 24.6 | 44.5 | 34.3 | 37.1 | 41.3 | 34.8 | 48.1 | 40.4 | 1 |
| domainpertinence | 2,202 | 35% | 0.563 | 66.0 | 7.8 | 51.0 | 24.3 | 44.6 | 34.4 | 37.0 | 41.2 | 34.8 | 48.1 | 40.4 | 2 |
| weirdness | 2,202 | 35% | 0.556 | 64.0 | 7.5 | 52.4 | 25.0 | 44.3 | 34.1 | 37.1 | 41.3 | 34.8 | 48.1 | 40.4 | 2 |
| topicmodeling | 1,437 | 38% | 0.375 | 45.0 | 5.3 | 35.4 | 16.9 | 36.6 | 28.2 | 37.7 | 35.7 | 37.7 | 34.0 | 35.7 | 1 |
| nmf | 2,202 | 35% | 0.453 | 42.0 | 5.0 | 45.4 | 21.7 | 46.5 | 35.8 | 36.3 | 40.4 | 34.8 | 48.1 | 40.4 | 1 |
| pmi | 1,294 | 36% | 0.471 | 41.0 | 4.8 | 48.0 | 22.9 | 39.8 | 30.7 | 36.4 | 32.6 | 36.4 | 29.5 | 32.6 | 1 |
| llr | 1,294 | 36% | 0.571 | 65.0 | 7.7 | 50.0 | 23.9 | 41.7 | 32.1 | 36.4 | 32.6 | 36.4 | 29.5 | 32.6 | 1 |
| chi2 | 1,294 | 36% | 0.534 | 56.0 | 6.6 | 49.4 | 23.6 | 40.8 | 31.4 | 36.4 | 32.6 | 36.4 | 29.5 | 32.6 | 1 |
| dice | 1,294 | 36% | 0.535 | 54.0 | 6.4 | 51.8 | 24.7 | 41.4 | 31.9 | 36.4 | 32.6 | 36.4 | 29.5 | 32.6 | 1 |
| tscore | 1,294 | 36% | 0.542 | 60.0 | 7.1 | 47.4 | 22.6 | 42.0 | 32.4 | 36.4 | 32.6 | 36.4 | 29.5 | 32.6 | 1 |
| yake | 2,186 | 35% | 0.373 | 34.0 | 4.0 | 36.8 | 17.6 | 36.2 | 27.9 | 34.2 | 38.0 | 34.6 | 47.4 | 40.0 | 3 |
| textrank | 2,202 | 35% | 0.290 | 27.0 | 3.2 | 28.4 | 13.5 | 32.4 | 25.0 | 36.0 | 40.1 | 34.8 | 48.1 | 40.4 | 1 |
| singlerank | 2,202 | 35% | 0.251 | 17.0 | 2.0 | 24.6 | 11.7 | 32.9 | 25.3 | 36.0 | 40.1 | 34.8 | 48.1 | 40.4 | 1 |
| positionrank | 2,202 | 35% | 0.228 | 12.0 | 1.4 | 25.2 | 12.0 | 31.0 | 23.9 | 35.9 | 39.9 | 34.8 | 48.1 | 40.4 | 1 |
| rake | 653 | 16% | 0.209 | 21.0 | 1.7 | 15.4 | 5.5 | 16.2 | 7.1 | 16.2 | 7.1 | 16.2 | 4.6 | 7.1 | 1 |
| roger | 3,873 | 15% | 0.290 | 29.0 | 2.4 | 24.8 | 8.8 | 19.2 | 11.6 | 18.9 | 17.5 | 15.0 | 25.1 | 18.8 | 31 |
| features | 4,876 | 32% | 0.564 | 66.0 | 5.5 | 56.2 | 19.9 | 52.3 | 31.5 | 44.1 | 40.8 | 32.1 | 67.5 | 43.5 | 8 |
| token-classification | 1,981 | 67% | 0.656 | 73.0 | 6.0 | 63.4 | 22.5 | 63.9 | 38.5 | 66.7 | 61.5 | 66.7 | 57.0 | 61.5 | 395 |

#### Observations

- **The reference corpus helps on RSDO5**, unlike on HTEC. With the other domains as
  reference, relevance, weirdness or domain pertinence has the best precision at the
  top of the ranking on every domain: P@100 of 87–88 on veterinary science, for
  example, against 69 for TF-IDF.
- **Only 42–54% of the lemmatised gold terms are candidates.** RSDO5 annotates every
  term, including single occurrences and terms that don't match the noun-phrase
  pattern, so that is the recall of the unsupervised methods over all their candidates.
- **NC-value** is a little ahead of C-value on every domain, with AvP 0.003 to 0.019
  higher, unlike on the English datasets and HTEC.
- **Token classification** is far ahead of every other method, as on ACTER and HTEC.
- **The features method** is second, with F1 43–52 over the candidates it classifies as
  terms, ahead of the 30–38 of Ljubešić et al. (2019). Its learned tag sequences cover
  more of the gold terms than the noun-phrase pattern, and its recall is 68–78.
- **Roger and RAKE** are the weakest: 15–19 and 7–14 F1 over all their candidates.

### Irish Wikipedia gold standard

#### Setup

The [Irish Wikipedia gold standard](https://doi.org/10.5281/zenodo.24132067) of McCrae &
Doyle (2019) has 11 Irish Wikipedia articles, 5,178 tokens and 577 distinct gold terms.
The two annotations that are a lone quotation mark are left out, as
`datasets.load("irish-wikipedia")` does.

- **Candidates.** The articles are tokenised but have no lemmas or tags, so, as on
  HTEC, words are read as written and there is no part-of-speech pattern. Candidates are
  sequences of up to four words that don't start or end with a word of stopwords-iso's
  Irish list (`stopwords="ga"`), whose words each have a letter or digit, and that occur
  at least twice. The unsupervised methods run on all 11 articles.
- **The reference corpus** is a sample of Irish Wikipedia from the Leipzig Corpora
  Collection, its largest Irish sample: 30,000 sentences, or 560,000 words. The 13
  sentences it shares with the gold standard are left out.
- **The supervised methods** are evaluated leaving one article out, since 11 articles
  are too few for a fixed split. Each article is the test article in turn: the methods
  train on nine of the others, and token classification keeps the best epoch on the
  remaining one, the next article. The features method's candidates are every
  sequence of up to four words that doesn't start or end with a stopword, including
  those that occur once, with the reference corpus below. The results of the 11 folds
  are pooled: the terms
  found in each test article are scored together against all the gold terms. Rankings
  from 11 models can't be merged, so the supervised methods have no ranked scores.
- **Cut-offs.** With 577 gold terms, precision, recall and F1 are reported for the top
  100 and 500 terms, and over all candidates.

#### Comparison with published results

McCrae & Doyle (2019, Table 4) report precision, recall and F1 for each of the B, I and
O tags of the tokens. Their tagger is a maximum-entropy model trained on other Irish
corpora, tagged by part of speech or by matching the Tearma term database. TermXtract's
token classification is pooled over the folds, as described above:

| Tag | Best published P | R | F1 | TermXtract P | R | F1 | |
|---|---:|---:|---:|---:|---:|---:|---|
| B | 70.7 | 50.6 | 59.0 | 76.6 | 75.7 | 76.2 | **better** |
| I | 74.7 | 46.8 | 57.6 | 84.8 | 58.2 | 69.0 | **better** |
| O | 90.7 | 83.5 | 86.9 | 91.1 | 95.8 | 93.4 | **better** |

The best published F1 for B is from the merged part-of-speech corpora converted to IOB
tags, for I from the merged part-of-speech corpora, and for O from all the corpora
together (Tearma alone ties). The comparison favours TermXtract: it fine-tunes
XLM-RoBERTa on articles from the gold standard itself, while McCrae & Doyle train on
other corpora. Per occurrence, TermXtract tags terms with precision 65.0, recall 64.2 and
F1 64.6.

#### Results

| Method | Candidates | Gold | AvP (ATR4S) | P@100 | F1@100 | P@500 | F1@500 | P | R | F1 | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tfidf | 638 | 20% | 0.424 | 30.0 | 8.9 | 20.8 | 19.3 | 19.9 | 22.0 | 20.9 | 0 |
| ridf | 638 | 20% | 0.313 | 29.0 | 8.6 | 23.2 | 21.5 | 19.9 | 22.0 | 20.9 | 0 |
| cvalue | 638 | 20% | 0.238 | 24.0 | 7.1 | 23.6 | 21.9 | 19.9 | 22.0 | 20.9 | 0 |
| ncvalue | 638 | 20% | 0.247 | 22.0 | 6.5 | 23.0 | 21.4 | 19.9 | 22.0 | 20.9 | 0 |
| basic | 638 | 20% | 0.206 | 15.0 | 4.4 | 19.8 | 18.4 | 19.9 | 22.0 | 20.9 | 0 |
| combobasic | 638 | 20% | 0.168 | 15.0 | 4.4 | 19.8 | 18.4 | 19.9 | 22.0 | 20.9 | 0 |
| domaincoherence | 638 | 20% | 0.217 | 19.0 | 5.6 | 21.4 | 19.9 | 19.9 | 22.0 | 20.9 | 0 |
| relevance | 638 | 20% | 0.302 | 25.0 | 7.4 | 23.2 | 21.5 | 19.9 | 22.0 | 20.9 | 1 |
| domainpertinence | 638 | 20% | 0.317 | 26.0 | 7.7 | 23.8 | 22.1 | 19.9 | 22.0 | 20.9 | 1 |
| weirdness | 638 | 20% | 0.315 | 25.0 | 7.4 | 23.8 | 22.1 | 19.9 | 22.0 | 20.9 | 1 |
| topicmodeling | 631 | 20% | 0.275 | 28.0 | 8.3 | 20.2 | 18.8 | 19.8 | 21.7 | 20.7 | 0 |
| nmf | 638 | 20% | 0.385 | 34.0 | 10.0 | 19.0 | 17.6 | 19.9 | 22.0 | 20.9 | 0 |
| pmi | 188 | 20% | 0.273 | 22.0 | 6.5 | 19.7 | 9.7 | 19.7 | 6.4 | 9.7 | 0 |
| llr | 188 | 20% | 0.280 | 19.0 | 5.6 | 19.7 | 9.7 | 19.7 | 6.4 | 9.7 | 0 |
| chi2 | 188 | 20% | 0.249 | 20.0 | 5.9 | 19.7 | 9.7 | 19.7 | 6.4 | 9.7 | 0 |
| dice | 188 | 20% | 0.243 | 20.0 | 5.9 | 19.7 | 9.7 | 19.7 | 6.4 | 9.7 | 0 |
| tscore | 188 | 20% | 0.283 | 20.0 | 5.9 | 19.7 | 9.7 | 19.7 | 6.4 | 9.7 | 0 |
| yake | 652 | 20% | 0.305 | 28.0 | 8.3 | 21.4 | 19.9 | 20.2 | 22.9 | 21.5 | 0 |
| textrank | 638 | 20% | 0.233 | 23.0 | 6.8 | 17.6 | 16.3 | 19.9 | 22.0 | 20.9 | 0 |
| singlerank | 638 | 20% | 0.152 | 15.0 | 4.4 | 18.2 | 16.9 | 19.9 | 22.0 | 20.9 | 0 |
| positionrank | 638 | 20% | 0.207 | 27.0 | 8.0 | 20.2 | 18.8 | 19.9 | 22.0 | 20.9 | 0 |
| rake | 42 | 60% | 0.759 | 59.5 | 8.1 | 59.5 | 8.1 | 59.5 | 4.3 | 8.1 | 0 |
| roger | 194 | 19% | – | – | – | – | – | 19.1 | 6.4 | 9.6 | 154 |
| features | 1,013 | 29% | – | – | – | – | – | 29.1 | 51.1 | 37.1 | 79 |
| token-classification | 575 | 66% | – | – | – | – | – | 65.7 | 65.5 | 65.6 | 192 |

#### Observations

- **Few gold terms are candidates.** Only 22% of the gold terms are candidates of the
  unsupervised methods. 435 of the 577 gold terms occur only once, and 20 are longer
  than four words. A few start with the article `an` or `na`, as in `an tSomáilis`,
  and both are stopwords.
- **The reference corpus helps a little.** Weirdness and domain pertinence have the best
  F1@500 (22.1), against 19.3 for TF-IDF. NMF and TF-IDF have the best precision at
  the top, with P@100 of 34 and 30.
- **Token classification** is far ahead of every other method, as on the other datasets,
  even with only nine articles to train on.
- **The features method** is second, with F1 37.1. Unlike the unsupervised methods, its
  candidates include those that occur once, so it finds 51% of the gold terms.
- **Roger and RAKE** are the weakest. RAKE keeps only 42 candidates, since its phrases
  are long and must occur twice.

## References

- Astrakhantsev, N. (2018). ATR4S: toolkit with state-of-the-art automatic terms
  recognition methods in Scala. *Language Resources and Evaluation*, 52, 853–872.
  https://arxiv.org/abs/1611.07804
- Lang, C., Wachowiak, L., Heinisch, B., & Gromann, D. (2021). Transforming term
  extraction: transformer-based approaches to multilingual term extraction across
  domains. In *Findings of ACL-IJCNLP 2021*, 3607–3620.
  https://aclanthology.org/2021.findings-acl.316
- Banerjee, S. (2025). *Deep learning for automatic term extraction* [Doctoral thesis,
  University of Galway].
- Banerjee, S., Chakravarthi, B. R., & McCrae, J. P. (2025). Benchmarking Hindi term
  extraction in education: a dataset and analysis. In *Proceedings of Language, Data
  and Knowledge (LDK 2025)*.
- Frantzi, K., Ananiadou, S., & Mima, H. (2000). Automatic recognition of multi-word
  terms: the C-value/NC-value method. *International Journal on Digital Libraries*,
  3(2), 115–130. https://doi.org/10.1007/s007999900023
- Boudin, F. (2016). pke: an open source python-based keyphrase extraction toolkit. In
  *Proceedings of COLING 2016: System Demonstrations*, 69–73.
  https://aclanthology.org/C16-2015
- Campos, R., Mangaravite, V., Pasquali, A., Jorge, A., Nunes, C., & Jatowt, A. (2020).
  YAKE! Keyword extraction from single documents using multiple local features.
  *Information Sciences*, 509, 257–289. https://doi.org/10.1016/j.ins.2019.09.013
- Florescu, C., & Caragea, C. (2017). PositionRank: an unsupervised approach to
  keyphrase extraction from scholarly documents. In *Proceedings of ACL 2017*,
  1105–1115. https://aclanthology.org/P17-1102
- Goldhahn, D., Eckart, T., & Quasthoff, U. (2012). Building large monolingual
  dictionaries at the Leipzig Corpora Collection: from 100 to 200 languages. In
  *Proceedings of LREC 2012*, 759–765.
- Ljubešić, N., Fišer, D., & Erjavec, T. (2019). KAS-term: extracting Slovene terms
  from doctoral theses via supervised machine learning. In *Proceedings of TSD 2019*,
  115–126.
- McCrae, J. P., & Doyle, A. (2019). Adapting term recognition to an under-resourced
  language: the case of Irish. In *Proceedings of the Celtic Language Technology
  Workshop*, 48–57. https://aclanthology.org/W19-6907
- Mihalcea, R., & Tarau, P. (2004). TextRank: bringing order into texts. In
  *Proceedings of EMNLP 2004*, 404–411. https://aclanthology.org/W04-3252
- Navigli, R., & Velardi, P. (2004). Learning domain ontologies from document
  warehouses and dedicated web sites. *Computational Linguistics*, 30(2), 151–179.
- Tran, H. T. H., Martinc, M., Repar, A., Doucet, A., & Pollak, S. (2022). A
  transformer-based sequence-labeling approach to the Slovenian cross-domain automatic
  term extraction. In *Proceedings of the Conference on Language Technologies and
  Digital Humanities (JTDH 2022)*, 196–204.
- Rigouts Terryn, A., Hoste, V., & Lefever, E. (2021). HAMLET: Hybrid Adaptable Machine
  Learning approach to Extract Terminology. *Terminology*, 27(2), 254–293.
  https://doi.org/10.1075/term.20017.rig
- Rigouts Terryn, A., Hoste, V., Drouin, P., & Lefever, E. (2020). TermEval 2020:
  shared task on automatic term extraction using the Annotated Corpora for Term
  Extraction Research (ACTER) dataset. In *Proceedings of CompuTerm 2020*, 85–94.
  https://aclanthology.org/2020.computerm-1.12
- Vivaldi, J., Màrquez, L., & Rodríguez, H. (2001). Improving term extraction by system
  combination using boosting. In *Machine Learning: ECML 2001*, LNCS 2167, 515–526.
- Wan, X., & Xiao, J. (2008). Single document keyphrase extraction using neighborhood
  knowledge. In *Proceedings of AAAI 2008*, 855–860.
