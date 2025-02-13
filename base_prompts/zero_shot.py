# Use a pipeline as a high-level helper
from transformers import pipeline, BitsAndBytesConfig
from dataset import *
from string import Template
from tqdm import tqdm
import json

gold_dataset = load_dataset_pkl("gold.pkl")

pipe = pipeline("text-generation", 
                model="chavinlo/gpt4-x-alpaca", 
                model_kwargs={
                    "quantization_config": BitsAndBytesConfig(True), 
                    "device_map": "auto"})

prompt = Template("""Below is an instruction that describes a task. Write a response that appropriately completes the request

### Instruction: Given a source article and evidence documents, generate the edited source article to incorporate new information from the evidence documents. Prefer substitutions and editing sentences over adding new ones. Just generate the updated article and not the evidences.

Source Article: $source

$evidences

### Response: Updated Article: """)


with open("gpt4-x-alpaca-gold-predictions.jsonl", "w") as f:
    for updated_article in tqdm(gold_dataset, desc="generating predictions", total=len(gold_dataset)):
        evidences = ((e.content, e.title, e.section) if not isinstance(e.content, Table) else (e.content.raw_text, e.title, e.section) for e in updated_article.evidences.values())
        evidence_text = "\n\n".join(f"Evidence {i + 1}:\nTitle: {title}\nSection: {section}\n{e}" for i, (e, title, section) in enumerate(evidences))

        input_prompt = prompt.substitute(source=updated_article.normalised_source, evidences=evidence_text) 
        generated_output: str = pipe(input_prompt, max_new_tokens=750)[0]["generated_text"]
        predicted_target = generated_output.removeprefix(input_prompt)
        print(json.dumps({"prediction": predicted_target,"source": updated_article.normalised_source , "id": str(updated_article.id), "target": updated_article.normalised_target}), file=f, flush=True)

