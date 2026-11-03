# Evaluation

`evaluate()` scores an `ATEResults` object against a gold standard. It is also available as `termxtract.evaluate`.

```python
from termxtract import TermExtractor, evaluate

corpus = [
    "Heart failure patients often have high blood pressure.",
    "Blood pressure and ejection fraction predict heart failure.",
]
gold = {"heart failure", "blood pressure", "ejection fraction"}

results = TermExtractor(method="cvalue", n=2).extract(corpus)
scores = evaluate(results, gold, ks=(3,))
print(f"P={scores.precision:.2f} R={scores.recall:.2f} F1={scores.f1:.2f}")
print(f"AvP={scores.average_precision:.2f} P@3={scores.precision_at_k[3]:.2f}")
# P=0.17 R=1.00 F1=0.29
# AvP=0.63 P@3=0.33

top3 = evaluate(results, gold, top_k=3)
print(f"P={top3.precision:.2f} R={top3.recall:.2f} MCC={top3.mcc:.2f}")
# P=0.33 R=0.33 MCC=0.20
```

Matthews correlation coefficient (`mcc`) treats the terms the extractor scored as the candidates, so it is informative only when `top_k` keeps part of the ranking.

::: termxtract.evaluation
    options:
      show_root_heading: false
      members:
        - evaluate
        - Evaluation
        - Gold
        - normalize_term
        - precision_at
        - average_precision
        - matthews_corrcoef
