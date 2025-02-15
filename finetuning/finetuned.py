from vllm import LLM, SamplingParams
from string import Template
from tqdm import tqdm
from dataset import *
import json

resume_from = 0

prompt = Template("""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a knowledgeable, efficient, and direct AI assistant. Provide concise answers, focusing on the key information needed. Engage in productive collaboration with the user.<|eot_id|><|start_header_id|>user<|end_header_id|>

Given a source article and evidence documents, edit the source article to incorporate new information from the evidence documents. Prefer substitutions and editing sentences over adding new ones. Just generate the updated article and not the evidences.
Source Article: $source

$evidences<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                      
Here's the source article updated with information from the given evidences incorporated.
Updated Article: """)

if __name__ == "__main__":
    split = "gold"
    
    gold_dataset = load_dataset_pkl(f"{split}.pkl")[resume_from:]
    sampling_params = SamplingParams(max_tokens=5000, temperature=0, min_tokens=10)

    llm = LLM(model="merged_model", dtype="half")

    
    with open(f"llama3-8b-finetuned-lora-predictions-{split}.jsonl", "w" if not resume_from else "a") as f:
        for updated_article in tqdm(gold_dataset, desc="generating predictions", total=len(gold_dataset)):
            evidences = ((e.content, e.title, e.section) if not isinstance(e.content, Table) else (e.content.raw_text, e.title, e.section) for e in updated_article.evidences.values())
            evidence_text = "\n\n".join(f"Evidence {i + 1}:\nTitle: {title}\nSection: {section}\n{e}" for i, (e, title, section) in enumerate(evidences))

            input_prompt = prompt.substitute(source=updated_article.normalised_source, evidences=evidence_text) 
            
            predicted_target = llm.generate(input_prompt, sampling_params)[0].outputs[0].text
            print(json.dumps({"prediction": predicted_target,"source": updated_article.normalised_source , "id": str(updated_article.id), "target": updated_article.normalised_target}), file=f, flush=True)
