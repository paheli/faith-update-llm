from dataset import *

train_dataset = load_dataset_pkl("/fs/scratch/rb_bd_dlp_rng-dl01_cr_AIM_employees/students/tty3kor/fruit/dataset/train.pkl")

length_bins = {}

def get_comparable_string(a: str):
    return a.strip().replace("\n", "").replace(" ", "")


def length_bin_transform(length: int):
    return length // 100

for article in train_dataset:
    if get_comparable_string(article.normalised_source) == get_comparable_string(article.normalised_target):
        continue 
    length_bins.setdefault(length_bin_transform(len(article.normalised_source)), []).append(article)

def get_closest_in_train_set(article: ArticleUpdate):
    length_hash = length_bin_transform(len(article.normalised_source))
    if length_hash not in length_bins:
        length_hash = sorted(length_bins.keys(), key=lambda x: abs(x - length_hash))[0]
    table_count = len([e for e in article.evidences.values() if isinstance(e.content, Table)])
    return sorted(length_bins[length_hash], key=lambda a : (abs(len([e for e in article.evidences.values() if isinstance(e.content, Table)]) - table_count),abs(len(article.evidences) - len(a.evidences))))
