from transformers import pipeline
from functools import reduce
import json
import os
from update_rouge import update_rouge, update_rouge_aggregate
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
from statistics import mean
from dataset import *
import sys
from pathlib import Path
from tqdm import tqdm

ner_pipe = pipeline("ner", model="dslim/bert-base-NER",
                    aggregation_strategy='average', device_map="cuda:0")


dataset = load_dataset_pkl(f"gold.pkl")
dataset = {str(i.id): i for i in dataset}


def argmin(values):
    return max(enumerate(values), key=lambda x: x[1])[0]


predictions, targets, sources = [], [], []
update_rouge1, update_rouge2, update_rougeLsum = [], [], []
entity_recalls = []
entity_precisions = []
unsupported_entities_scores = []

source_target_similarity, source_prediction_similarity = [], []

score_type = "update_rougeLsum"

evidence_count_scores = {}


def get_comparable_string(a: str):
    return a.strip().replace("\n", "").replace(" ", "")


similarity_score_type = "rougeLsum"
scorer_rouge = rouge_scorer.RougeScorer([similarity_score_type])

copies = 0
necessary_copies = 0

appends = []
unnecessary_appends = []

table_percentages_scores = {}

required_edit_lengths = {}

evidences_per_update = {}

file_name = "llama3-8b-predictions-filter-rank-random-gold.jsonl"

folder_path = Path(file_name).stem
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

sys.stdout = open(f"{folder_path}/metrics.txt", "w")


with open(file_name) as f:
    for line in tqdm(f, desc="evaluating", total=len(dataset)):
        row = json.loads(line)
        source_target_similarity.append(scorer_rouge.score(
            row["source"], row["target"])[similarity_score_type].fmeasure)
        source_prediction_similarity.append(scorer_rouge.score(
            row["source"], row["prediction"])[similarity_score_type].fmeasure)
        article = dataset[row["id"]]
        recall = sum(1 for e in article.target_entities if article.normalised_target[e['start']: e['end']].lower(
        ).strip() in row["prediction"].lower()) / len(article.target_entities)
        predictions.append(row["prediction"])
        targets.append(row["target"])
        sources.append(row["source"])
        entity_recalls.append(recall)
        score = update_rouge(row["source"], row["target"], row["prediction"])
        update_rouge1.append(score["update_rouge1"].fmeasure)
        update_rouge2.append(score["update_rouge2"].fmeasure)
        update_rougeLsum.append(score["update_rougeLsum"].fmeasure)
        evidence_count_scores.setdefault(len(article.evidences), [])
        evidence_count_scores[len(article.evidences)].append(
            score[score_type].fmeasure)

        entities = {e["word"].lower().strip()
                    for e in ner_pipe(row["prediction"])}  # type: ignore
        target_entities = {e["word"].lower().strip()
                           for e in ner_pipe(row["target"])}  # type: ignore
        # target_entities = {article.normalised_target[e['start']: e['end']].lower().strip() for e in article.target_entities}

        precision = len(entities & target_entities) / \
            len(entities) if len(entities) else 1
        entity_precisions.append(precision)

        # supporting_entities = {article.normalised_source[e['start']: e['end']].lower().strip() for e in article.source_entities}
        supporting_entities = {e["word"].lower().strip()
                               for e in ner_pipe(row["source"])}  # type: ignore
        texts = []
        for evidence in article.evidences.values():
            text = evidence.content.raw_text if isinstance(
                evidence.content, Table) else evidence.content
            texts.append(f"{evidence.title} {evidence.section} {text}")

        for b in ner_pipe(texts, batch_size=48):  # type: ignore
            supporting_entities |= {e["word"].lower().strip()
                                    for e in b}  # type: ignore

            # for e in evidence.entities:
            # supporting_entities.add(evidence.content.raw_text[e['start']:e['end']].lower().strip() if isinstance(evidence.content, Table) else evidence.content[e["start"]: e['end']].lower().strip())
        unsupported_entities = len(entities - supporting_entities)

        unsupported_entities_scores.append(unsupported_entities)
        if get_comparable_string(row["source"]) == get_comparable_string(row["target"]):
            necessary_copies += 1

        if get_comparable_string(row["source"]) == get_comparable_string(row["prediction"]):
            copies += 1
        elif get_comparable_string(row["prediction"]).startswith(get_comparable_string(row["source"])):
            appends.append(score[score_type].fmeasure)
            if not get_comparable_string(row["target"]).startswith(get_comparable_string(row["source"])):
                unnecessary_appends.append(score[score_type].fmeasure)
        table_percentage = len([e for e in article.evidences.values(
        ) if isinstance(e.content, Table)]) / len(article.evidences)
        table_percentages_scores.setdefault(table_percentage, [])
        table_percentages_scores[table_percentage].append(
            score[score_type].fmeasure)

        required_edit_length = (len(article.normalised_target) -
                                len(article.normalised_source)) / len(article.normalised_source)
        required_edit_lengths.setdefault(required_edit_length, [])
        required_edit_lengths[required_edit_length].append(
            score[score_type].fmeasure)

        updated_sentences_count = sum(
            1 for s in article.updated_article_sentences if isinstance(s, UpdatedSentence))
        if updated_sentences_count:
            references_per_sentence = sum(len(s.evidence_indices) for s in article.updated_article_sentences if isinstance(
                s, UpdatedSentence)) / updated_sentences_count
            evidences_per_update.setdefault(references_per_sentence, [])
            evidences_per_update[references_per_sentence].append(
                score[score_type].fmeasure)


aggregate_scores = update_rouge_aggregate(sources, targets, predictions)

table_only_scores = table_percentages_scores[1]
mixed_scores = reduce(lambda a, b: a + b, (v for k,
                      v in table_percentages_scores.items() if k not in [0, 1]), [])
plain_text_only_scores = table_percentages_scores[0]

print(f"{aggregate_scores=}")
for individual_score_type, individual_score in [("update_rouge1", update_rouge1),
                                                ("update_rouge2", update_rouge2),
                                                ("update_rougeLsum",
                                                 update_rougeLsum),
                                                ("entity_recall", entity_recalls),
                                                ("entity_precision",
                                                 entity_precisions),
                                                ("unsupported_entities",
                                                 unsupported_entities_scores),
                                                (f"source_prediction_similarity_{similarity_score_type}", source_prediction_similarity),
                                                (f"source_target_similarity_{similarity_score_type}", source_target_similarity),
                                                (f"table_only_{score_type}",
                                                 table_only_scores),
                                                (f"mixed_{score_type}",
                                                 mixed_scores),
                                                (f"text_only_{score_type}",
                                                 plain_text_only_scores),
                                                (f"appends_{score_type}",
                                                 appends),
                                                (f"unnecessary_appends_{score_type}", unnecessary_appends)]:
    # Adjust bins for desired granularity
    plt.hist(individual_score, edgecolor='black', bins='auto')
    plt.title(individual_score_type)
    plt.xlabel(individual_score_type)
    plt.ylabel('Frequency')
    plt.grid(True)

    if individual_score_type in [f"source_prediction_similarity_{similarity_score_type}", f"source_target_similarity_{similarity_score_type}"]:
        plt.xlim(0, 1)
        plt.ylim(0, 300)

    plt.savefig(f"{folder_path}/{individual_score_type}.png", transparent=True)
    plt.clf()

for label, individual_score_type, individual_score in [("Evidence Count", f"Mean {score_type}", evidence_count_scores),
                                                       ("Tabular Evidence Percentage",
                                                        f"Mean {score_type}", table_percentages_scores),
                                                       ("Required Edit Length",
                                                        f"Mean {score_type}", required_edit_lengths),
                                                       ("References per Updated Sentences", f"Mean {score_type}", evidences_per_update)]:
    items = sorted(list(individual_score.items()), key=lambda x: x[0])
    plt.plot([i[0] for i in items], [mean(i[1]) for i in items])
    plt.title(f"{label} vs {individual_score_type}")
    plt.xlabel(label)
    plt.ylabel(individual_score_type)
    plt.grid(True)
    plt.savefig(f"{folder_path}/{label}.png", transparent=True)
    plt.clf()


print(f"{copies=}")
print(f"{necessary_copies=}")

print(f"{mean(entity_recalls)=}")
print(f"{mean(entity_precisions)=}")
print(f"{mean(unsupported_entities_scores)=}")

print(f"{mean(source_target_similarity)=}")

print(f"{mean(source_prediction_similarity)=}")

print(f"{mean(plain_text_only_scores)=}")
print(f"{mean(mixed_scores) =}")
print(f"{mean(table_only_scores)=}")

print(f"{len(appends)=}")
print(f"{mean(appends)=}")

print(f"{len(unnecessary_appends)=}")
print(f"{mean(unnecessary_appends)=}")
