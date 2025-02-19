from vllm import LLM, SamplingParams
from string import Template
from tqdm import tqdm
from dataset import *
import json

resume_from = 0

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

if __name__ == "__main__":
    split = "gold"
    
    gold_dataset = load_dataset_pkl(f"{split}.pkl")[resume_from:]
    sampling_params = SamplingParams(max_tokens=5000, temperature=0, min_tokens=10)

    llm = LLM(model="unsloth/llama-3-8b-Instruct", dtype="half")

    prompt = Template("""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a knowledgeable, efficient, and direct AI assistant. Provide concise answers, focusing on the key information needed. Engage in productive collaboration with the user.<|eot_id|><|start_header_id|>user<|end_header_id|>

Given a source article and evidence documents, edit the source article to incorporate new information from the evidence documents. Prefer substitutions and editing sentences over adding new ones. Just generate the updated article and not the evidences.
Source Article: $source

$evidences<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                      
Here's the source article updated with information from the given evidences incorporated.
Updated Article: """)
    with open(f"llama3-8b-predictions-streamed-{split}.jsonl", "w" if not resume_from else "a") as f:
        for updated_article in tqdm(gold_dataset, desc="generating predictions", total=len(gold_dataset)):
            source = updated_article.normalised_source
            intermediate_predictions = []
            for evidence_batch in batch(list(updated_article.evidences.values()), 3):
                evidences = ((e.content, e.title, e.section) if not isinstance(e.content, Table) else (e.content.raw_text, e.title, e.section) for e in evidence_batch)
                evidence_text = "\n\n".join(f"Evidence {i + 1}:\nTitle: {title}\nSection: {section}\n{e}" for i, (e, title, section) in enumerate(evidences))

                input_prompt = prompt.substitute(source=source, evidences=evidence_text) 
                
                predicted_target = llm.generate(input_prompt, sampling_params)[0].outputs[0].text
                source = predicted_target
                intermediate_predictions.append(predicted_target)
            intermediate_predictions.pop() # last one goes in prediction
            print(json.dumps({"prediction": predicted_target,"source": updated_article.normalised_source , "id": str(updated_article.id), "target": updated_article.normalised_target, "intermediate_predictions": intermediate_predictions}), file=f, flush=True)
