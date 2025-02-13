from vllm import LLM, SamplingParams
from string import Template
from tqdm import tqdm
from dataset import *
import json

if __name__ == "__main__":
    continue_from = 0

    dataset = load_dataset_pkl("test-sampled.pkl")
    sampling_params = SamplingParams(max_tokens=5000, temperature=0, min_tokens=10)

    tiger_evaluations: dict[str, str] = {}
    no_errors: dict[str, bool] = {}
    with open("tiger_score-llama3-8b-test-sampled-evaluations.jsonl") as f:
        for row in f:
            row = json.loads(row)
            tiger_evaluations[row["id"]] = json.dumps(row["evaluation"])
            no_errors[row["id"]] = row["evaluation"]["num_errors"] == 0
    
    first_predictions: dict[str, str] = {}
    with open("llama3-8b-predictions-test-sampled.jsonl") as f:
        for row in f:
            row = json.loads(row)
            first_predictions[row["id"]] = row["prediction"]

    llm = LLM(model="unsloth/llama-3-8b-Instruct", dtype="half", tensor_parallel_size=4)

    step_two_prompt = Template("""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a knowledgeable, efficient, and direct AI assistant. Provide concise answers, focusing on the key information needed. Engage in productive collaboration with the user.<|eot_id|><|start_header_id|>user<|end_header_id|>

Given a source article and evidence documents, edit the source article to incorporate new information from the evidence documents. Prefer substitutions and editing sentences over adding new ones. Just generate the updated article and not the evidences.
Source Article: $source

$evidences<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                      
Here's the source article updated with information from the given evidences incorporated.
Updated Article: $updated_one<|eot_id|><|start_header_id|>user<|end_header_id|>

Given the following evaluation of the updated article in json format. Correct the errors and generate a new updated article.
Evaluation: $evaluation<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Here's the new updated article generated after taking into account the evaluation.
Updated Article: """) 
    with open("llama3-8b-test-sampled-predictions_tiger_correction.jsonl", "w" if not continue_from else "a") as f:
        for updated_article in tqdm(dataset[continue_from:], desc="generating predictions", total=len(dataset[continue_from:])):
            evidences = ((e.content, e.title, e.section) if not isinstance(e.content, Table) else (e.content.raw_text, e.title, e.section) for e in updated_article.evidences.values())
            evidence_text = "\n\n".join(f"Evidence {i + 1}:\nTitle: {title}\nSection: {section}\n{e}" for i, (e, title, section) in enumerate(evidences))
            first_prediction = first_predictions[str(updated_article.id)]
            if no_errors[str(updated_article.id)]:
                print(json.dumps({"prediction": first_prediction,"source": updated_article.normalised_source , "id": str(updated_article.id), "target": updated_article.normalised_target}), file=f, flush=True)
                continue
            input_prompt = step_two_prompt.substitute(source=updated_article.normalised_source, evidences=evidence_text, updated_one=first_prediction, evaluation=f"```json\n{tiger_evaluations[str(updated_article.id)]}\n```")
            predicted_target = llm.generate(input_prompt, sampling_params, use_tqdm=False)[0].outputs[0].text
            print(json.dumps({"prediction": predicted_target,"source": updated_article.normalised_source , "id": str(updated_article.id), "target": updated_article.normalised_target}), file=f, flush=True)
