from vllm import LLM, SamplingParams
from string import Template
from tqdm import tqdm
from dataset import *
import json

if __name__ == "__main__":
    gold_dataset = load_dataset_pkl("gold.pkl")
    sampling_params = SamplingParams(max_tokens=5000, temperature=0, min_tokens=10)

    llm = LLM(model="unsloth/llama-3-8b-Instruct", tensor_parallel_size=4, dtype="half")

    step_one_prompt = Template("""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a knowledgeable, efficient, and direct AI assistant. Provide concise answers, focusing on the key information needed. Engage in productive collaboration with the user.<|eot_id|><|start_header_id|>user<|end_header_id|>

Given a source article and evidence documents, find cases where information given in the evidences does not agree with the source article or where the source article does not contain information it should from the evidences. Generate the evidence number and the disagreement/missing information in the source article.
Source Article: $source

$evidences<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                      
Here's a list of cases where the evidences do not agree with the source article with both the evidence number and the disagreements.
Disagreements/Missing Information: """)
    step_two_prompt = Template("""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a knowledgeable, efficient, and direct AI assistant. Provide concise answers, focusing on the key information needed. Engage in productive collaboration with the user.<|eot_id|><|start_header_id|>user<|end_header_id|>

Given a source article and evidence documents, find cases where information given in the evidences does not agree with the source article or where the source article does not contain information it should from the evidences. Generate the evidence number and the disagreement/missing information in the source article.
Source Article: $source

$evidences<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                      
Here's a list of cases where the evidences do not agree with the source article with both the evidence number and the disagreements.
Disagreements/Missing Information: $disagreements<|eot_id|><|start_header_id|>user<|end_header_id|>

Now, given the above source article, evidence documents and the disagreements / missing information, edit the source article to incorporate new information by updating or adding from the evidence documents to correct the disagreements or add the missing information. Prefer substitutions and editing sentences over adding new ones. Just generate the updated article and not the evidences.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                      
Here's the source article updated with information from the given evidences incorporated.
Updated Article: """) 
    with open("llama3-8b-predictions_cot.jsonl", "w") as f:
        for updated_article in tqdm(gold_dataset, desc="generating predictions", total=len(gold_dataset)):
            evidences = ((e.content, e.title, e.section) if not isinstance(e.content, Table) else (e.content.raw_text, e.title, e.section) for e in updated_article.evidences.values())
            evidence_text = "\n\n".join(f"Evidence {i + 1}:\nTitle: {title}\nSection: {section}\n{e}" for i, (e, title, section) in enumerate(evidences))

            input_prompt = step_one_prompt.substitute(source=updated_article.normalised_source, evidences=evidence_text) 
            
            disagreements = llm.generate(input_prompt, sampling_params)[0].outputs[0].text
            input_prompt = step_two_prompt.substitute(source=updated_article.normalised_source, evidences=evidence_text, disagreements=disagreements)
            predicted_target = llm.generate(input_prompt, sampling_params)[0].outputs[0].text
            print(json.dumps({"prediction": predicted_target,"source": updated_article.normalised_source , "id": str(updated_article.id), "target": updated_article.normalised_target, "reasoning_step": disagreements}), file=f, flush=True)
