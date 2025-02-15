# Absolute Grading: Outputs score of 1 to 5

from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE
from string import Template
import json
from dataset import *

model = VLLM(model="prometheus-eval/prometheus-7b-v2.0", dtype="half")
judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT)
input_template = Template("Given a source article and evidence documents, edit the source article to incorporate new information from the evidence documents. Prefer substitutions and editing sentences over adding new ones. Just generate the updated article and not the evidences.\n\nSource Article: $source\n\n$evidences")

rubric_data = {
  "criteria":"Is the model proficient in updating articles based on new evidence, making correct and precise edits in place wherever possible?",
  "score1_description":"The model neglects to identify or incorporate new evidence into the article, resulting in outdated and inaccurate information.",
  "score2_description":"The model intermittently acknowledges new evidence but often fails to make correct and precise edits in place, leading to incomplete or inaccurate updates.",
  "score3_description":"The model typically identifies new evidence and attempts to make correct and precise edits in place, yet the updates might sometimes miss important details or lack precision.",
  "score4_description":"The model consistently identifies and incorporates new evidence into the article, making correct and precise edits in place. Nonetheless, there may still be sporadic oversights or deficiencies in the accuracy and precision of the updates.",
  "score5_description":"The model excels in identifying and incorporating new evidence into the article, persistently making correct and precise edits in place that demonstrate a thorough understanding of the subject matter. The updates are accurate, precise, and comprehensive, leaving no room for inaccuracies or incomplete information."
}

score_rubric = SCORE_RUBRIC_TEMPLATE.format(**rubric_data)

split = "test-sampled"
    
dataset = load_dataset_pkl(f"{split}.pkl")
dataset = {str(i.id): i for i in dataset}
instructions = []
responses = []
ids = []
with open("llama3-8b-predictions-test-sampled.jsonl") as f:
    for row in f:
        row = json.loads(row)
        ids.append(row["id"])
        updated_article = dataset[row["id"]]
        evidences = ((e.content, e.title, e.section) if not isinstance(e.content, Table) else (e.content.raw_text, e.title, e.section) for e in updated_article.evidences.values())
        evidence_text = "\n\n".join(f"Evidence {i + 1}:\nTitle: {title}\nSection: {section}\n{e}" for i, (e, title, section) in enumerate(evidences))

        input_context = input_template.substitute(source=updated_article.normalised_source, evidences=evidence_text)
        output = row["prediction"]
        instructions.append(input_context)
        responses.append(output)



feedbacks, scores = judge.absolute_grade(
    instructions=instructions,
    responses=responses,
    rubric=score_rubric,
)

with open("prometheus2-7b-llama3-8b-test-sampled-evaluations.jsonl", "w") as f:
    for article_id, feedback, score in zip(ids, feedbacks, scores):
        print(json.dumps({"id": article_id, "evaluation": {"score": score, "feedback": feedback}}), file=f)
