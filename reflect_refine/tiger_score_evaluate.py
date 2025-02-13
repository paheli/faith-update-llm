from tigerscore import TIGERScorer
from dataset import *
import json
from string import Template

instruction = "Given a source article and evidence documents, edit the source article to incorporate new information from the evidence documents. Prefer substitutions and editing sentences over adding new ones. Just generate the updated article and not the evidences."
input_template = Template("Source Article: $source\n\n$evidences")

if __name__ == "__main__":

    scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-7B", use_vllm=True)
    
    split = "gold"
    
    dataset = load_dataset_pkl(f"{split}.pkl")
    dataset = {str(i.id): i for i in dataset}
    input_contexts = []
    hypo_outputs = []
    ids = []
    with open("llama3-8b-predictions_run_2.jsonl") as f:
        for row in f:
            row = json.loads(row)
            ids.append(row["id"])
            updated_article = dataset[row["id"]]
            evidences = ((e.content, e.title, e.section) if not isinstance(e.content, Table) else (e.content.raw_text, e.title, e.section) for e in updated_article.evidences.values())
            evidence_text = "\n\n".join(f"Evidence {i + 1}:\nTitle: {title}\nSection: {section}\n{e}" for i, (e, title, section) in enumerate(evidences))

            input_context = input_template.substitute(source=updated_article.normalised_source, evidences=evidence_text)
            output = row["prediction"]
            input_contexts.append(input_context)
            hypo_outputs.append(output)



    results = scorer.score([instruction] * len(hypo_outputs), hypo_outputs, input_contexts)
    with open("tiger_score-llama3-8b-gold-evaluations.jsonl", "w") as f:
        for row_id, result in zip(ids, results):
            print(json.dumps({"id": row_id, "evaluation": result}), file=f, flush=True)