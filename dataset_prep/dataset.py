from collections import OrderedDict
from typing import Any, OrderedDict as OrderedDictType, TypedDict, Union
import re
from dataclasses import dataclass
import pickle


@dataclass(frozen=True, unsafe_hash=True)
class SentenceIndex:
    key: str

    def __str__(self) -> str:
        return self.key


@dataclass(frozen=True, unsafe_hash=True)
class EvidenceIndex:
    key: str

    def __str__(self) -> str:
        return self.key


@dataclass(frozen=True, unsafe_hash=True)
class UpdatedSentence:
    evidence_indices: list[EvidenceIndex]
    updated_text: str

    def __str__(self) -> str:
        return f"{' '.join(str(i) for i in self.evidence_indices)}, {self.updated_text}"


@dataclass(unsafe_hash=True)
class Table:
    raw_text: str
    caption: Union[str, None]
    header: Union[list[str], None]
    rows: list[list[str]]

    def __init__(self, raw_text):
        self.raw_text = raw_text
        row_delimiter = re.compile(r"\[ROW\]")
        rows = [i.strip() for i in row_delimiter.split(self.raw_text)]
        match = re.search(
            r"^(?:\[CAPTION\](.*?))?((\[HEADER\] .*?)?(?:(?:\[COL\].*?)+)?)?$", rows[0], re.DOTALL)
        if match is None:
            raise ValueError(f"{raw_text} is not a Table.")
        self.caption = match.group(1)
        if match.group(3) is not None:
            header = re.sub(r"\[HEADER\] ", "", match.group(2))
            rows.pop(0)
        else:
            header = None
            if match.group(2) is not None:
                rows[0] = match.group(2)
            else:
                rows.pop(0)

        self.header = Table.process_raw_row(
            header) if header is not None else None
        self.rows = [Table.process_raw_row(row) for row in rows]

    @staticmethod
    def process_raw_row(text: str) -> list[str]:
        return text.split("[COL] ")[1:]


class EntityType(TypedDict):
    id: str
    start: int
    end: int


@dataclass(unsafe_hash=True)
class Evidence:
    title: str
    section: str
    content: Union[Table, str]
    entities: list[EntityType]

    def __init__(self, title: str, section: str, text: str, entities: list[EntityType]):
        self.title = title
        self.section = section
        self.entities = entities
        self.content = Table(text) if "[COL]" in text else text


class ArticleUpdate:
    def __init__(self, id: Any, inputs: str, targets: str, json_data: dict[str, Any]):
        self.raw_inputs = inputs
        self.raw_targets = targets
        self.id = id
        self.normalised_source = json_data["source_article"]["text"]
        self.normalised_target = json_data["target_article"]["text"]
        self.source_entities: list[EntityType] = json_data["source_article"]["entities"]
        self.target_entities: list[EntityType] = json_data["target_article"]["entities"]
        self.new_entities: list[EntityType] = json_data["target_article"]["added_entities"]
        self.original_article, contexts = self.raw_inputs.split("[CONTEXT]")

        original_sentences = []
        article = self.original_article
        i = 0
        while True:
            sep_pattern = rf" ?(\[{i}\]) ?"
            i += 1
            splits = re.split(sep_pattern, article, 1)
            if len(splits) == 1:
                original_sentences.append(splits[0].strip())
                break
            else:
                original_sentences.extend(i.strip() for i in splits[:-1] if i)
                article = splits[-1]

        self.original_sentences: OrderedDictType[SentenceIndex, str] = OrderedDict(zip((SentenceIndex(k) for k in original_sentences[::2]),
                                                                                       original_sentences[1::2]))

        original_article_sep_regex = re.compile(
            f" ?({'|'.join(re.escape(i.key) for i in self.original_sentences.keys())}) ?")
        # remaining_context = contexts
        # contexts = []
        # i = 0
        # while True:
        #     sep_pattern = rf" ?(\({i}\)) ?"
        #     i += 1
        #     splits = re.split(sep_pattern, remaining_context, 1)
        #     if len(splits) == 1:
        #         contexts.append(splits[0].strip())
        #         break
        #     else:
        #         contexts.extend(i.strip() for i in splits[:-1] if i)
        #         remaining_context = splits[-1]

        self.evidences: OrderedDictType[EvidenceIndex, Evidence] = OrderedDict(
            (
                (EvidenceIndex(f'({i})'),
                 Evidence(e["mention"]["title"], e["mention"]["section"],
                          e["mention"]["text"], e["mention"]["entities"])
                 )
                for i, e
                in enumerate(json_data["annotated_mentions"])

                if f'{e["mention"]["title"]} {e["mention"]["section"]} {e["mention"]["text"]}' in contexts
            )
        )

        context_sep_regex = re.compile(
            f" ?({'|'.join(re.escape(i.key) for i in self.evidences)}) ?")
        updated_article_sentences: list[Union[str, SentenceIndex]] = [SentenceIndex(i) if re.match(
            original_article_sep_regex, i) else i for i in re.split(original_article_sep_regex, targets) if i]

        self.updated_article_sentences: list[Union[SentenceIndex, UpdatedSentence]] = [
        ]
        for sentence in updated_article_sentences:
            if isinstance(sentence, SentenceIndex):
                self.updated_article_sentences.append(sentence)
                continue

            evidences: list[EvidenceIndex] = []
            for i in context_sep_regex.split(sentence):
                if not i:
                    continue

                if context_sep_regex.match(i):
                    evidences.append(EvidenceIndex(i))

                else:
                    self.updated_article_sentences.append(
                        UpdatedSentence(evidences, i))
                    evidences = []


def load_dataset_pkl(path: str) -> list[ArticleUpdate]:
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    from tfrecord import tfrecord_loader
    import json
    from pathlib import Path
    from tqdm import tqdm
    from typing import Literal, Union
    split: Union[Literal["gold"], Literal["train"], Literal["test"]] = "test"

    test_folder_path = "test"
    train_folder_path = "train"
    gold_folder_path = "gold_test"

    folder_path = train_folder_path if split != "gold" else test_folder_path
    data = {}
    json_count = 0
    jsonl_files = list(Path(folder_path).glob('**/*.jsonl-*-of-*'))
    for p in tqdm(jsonl_files, total=len(jsonl_files), desc="Loading jsonl files"):
        with open(p) as f:
            for line in f:
                if not line:
                    continue
                json_count += 1
                row = json.loads(line)
                data[row["source_article"]["id"]] = [row, None]

    tfrecord_files = list(Path(folder_path).glob('**/*.tfrecords-*-of-*')
                          ) if split != "gold" else list(Path(gold_folder_path).glob('**/*.tfrecords'))
    for p in tqdm(tfrecord_files, total=len(tfrecord_files), desc="Loading tfrecords"):
        d = tfrecord_loader(str(p), None)
        for i in d:
            try:
                data[i["id"][0]][1] = i  # type: ignore
            except KeyError:
                data[i["id"][0]] = [None, i]  # type: ignore

    updates = []
    for text_id, (json_data, tf_data) in tqdm(data.items(), total=len(data), desc="parsing data"):
        if None in [json_data, tf_data]:
            continue
        id = tf_data["id"][0]
        inputs = tf_data["inputs"].decode()
        targets = tf_data["targets"].decode()

        update = ArticleUpdate(id, inputs, targets, json_data)
        updates.append(update)
    with open(f"{split}.pkl", "wb") as f:
        pickle.dump(updates, f)
