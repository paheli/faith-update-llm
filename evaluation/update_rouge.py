from rouge_score import rouge_scorer, scoring
import re

def extract_additions(source, target):
    normalized_additions = []
    for match in re.finditer(r"[^.]+\.?", target):
        if match.group(0) not in source:
            normalized_additions.append(match.group(0).strip())
    return normalized_additions

def update_rouge(input_text: str, target: str, prediction: str):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"])
    target_additions = " ".join(extract_additions(input_text, target))
    preduction_additions = " ".join(extract_additions(input_text, prediction))
    addition_scores = scorer.score(target=target_additions, prediction=preduction_additions)
    if target_additions.strip() or preduction_additions.strip():
        return {f"update_{k}": v for k,v in addition_scores.items()}
    else:
        return {f"update_{k}": scoring.Score(100, 100, 100) for k in addition_scores}

def update_rouge_aggregate(input_texts: list[str], targets: list[str], predictions: list[str]):
    aggregator = scoring.BootstrapAggregator()
    for i, t, p in zip(input_texts, targets, predictions):
        aggregator.add_scores(update_rouge(i, t, p))
    
    result = aggregator.aggregate()
    return {k: v.mid.fmeasure * 100 for k, v in result.items()}

