from vllm import LLM, SamplingParams
from string import Template
from tqdm import tqdm
from dataset import *
import json

if __name__ == "__main__":
    continue_from = 30

    gold_dataset = load_dataset_pkl("test-sampled.pkl")
    sampling_params = SamplingParams(max_tokens=5000, temperature=0, min_tokens=10)

    llm = LLM(model="unsloth/llama-3-8b-Instruct", tensor_parallel_size=4, dtype="half", enable_prefix_caching=True)

    step_one_prompt = Template("""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a knowledgeable, efficient, and direct AI assistant. Provide concise answers, focusing on the key information needed. Engage in productive collaboration with the user.<|eot_id|><|start_header_id|>user<|end_header_id|>

Given a source article and evidence documents, edit the source article to incorporate new information from the evidence documents. Prefer substitutions and editing sentences over adding new ones. Just generate the updated article and not the evidences.
Source Article: $source

$evidences<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                      
Here's the source article updated with information from the given evidences incorporated.
Updated Article: """)
    step_two_prompt = Template("""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a knowledgeable, efficient, and direct AI assistant. Provide concise answers, focusing on the key information needed. Engage in productive collaboration with the user.<|eot_id|><|start_header_id|>user<|end_header_id|>

Given a source article and evidence documents, edit the source article to incorporate new information from the evidence documents. Prefer substitutions and editing sentences over adding new ones. Just generate the updated article and not the evidences.
Source Article: $source

$evidences<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                      
Here's the source article updated with information from the given evidences incorporated.
Updated Article: $updated_one<|eot_id|><|start_header_id|>user<|end_header_id|>

Now, evaluate the generated updated article. Find cases where the updated article does not agree or contains missing information when compared with the supplied evidences. List out all such discrepancies with the evidence number and the reason.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Here's an evaluation of the updated article when compared to the evidences. The list of discrepancies is as follows:
""")
    step_three_prompt = Template("""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a knowledgeable, efficient, and direct AI assistant. Provide concise answers, focusing on the key information needed. Engage in productive collaboration with the user.<|eot_id|><|start_header_id|>user<|end_header_id|>

Given a source article and evidence documents, edit the source article to incorporate new information from the evidence documents. Prefer substitutions and editing sentences over adding new ones. Just generate the updated article and not the evidences.
Source Article: $source

$evidences<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                      
Here's the source article updated with information from the given evidences incorporated.
Updated Article: $updated_one<|eot_id|><|start_header_id|>user<|end_header_id|>

Now, evaluate the generated updated article. Find cases where the updated article does not agree or contains missing information when compared with the supplied evidences. List out all such discrepancies with the evidence number and the reason.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Here's an evaluation of the updated article when compared to the evidences. The list of discrepancies is as follows:
$discrepancies<|eot_id|><|start_header_id|>user<|end_header_id|>

Now, given the evaluation and the previously updated article, fix the discrepancies and generate a new updated article. Prefer substitutions and editing sentences over adding new ones.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Here's the updated article updated with information from the evidences and the evaluation done on the previous updated article. The discrepancies have been addressed in the following article.
Updated Article: """) 
    with open("llama3-8b-test-sampled-predictions_self_correction.jsonl", "w" if not continue_from else "a") as f:
        for updated_article in tqdm(gold_dataset[continue_from:], desc="generating predictions", total=len(gold_dataset[continue_from:])):
            evidences = ((e.content, e.title, e.section) if not isinstance(e.content, Table) else (e.content.raw_text, e.title, e.section) for e in updated_article.evidences.values())
            evidence_text = "\n\n".join(f"Evidence {i + 1}:\nTitle: {title}\nSection: {section}\n{e}" for i, (e, title, section) in enumerate(evidences))

            input_prompt = step_one_prompt.substitute(source=updated_article.normalised_source, evidences=evidence_text) 
            updated_one = llm.generate(input_prompt, sampling_params, use_tqdm=False)[0].outputs[0].text
            input_prompt = step_two_prompt.substitute(source=updated_article.normalised_source, evidences=evidence_text, updated_one=updated_one)
            discrepancies = llm.generate(input_prompt, sampling_params, use_tqdm=False)[0].outputs[0].text
            input_prompt = step_three_prompt.substitute(source=updated_article.normalised_source, evidences=evidence_text, updated_one=updated_one, discrepancies=discrepancies)
            predicted_target = llm.generate(input_prompt, sampling_params, use_tqdm=False)[0].outputs[0].text
            print(json.dumps({"prediction": predicted_target,"source": updated_article.normalised_source , "id": str(updated_article.id), "target": updated_article.normalised_target, "initial_updated": updated_one, "discrepancies": discrepancies}), file=f, flush=True)
