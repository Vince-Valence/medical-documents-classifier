import json
from collections import namedtuple

WordTuple = namedtuple("WordTuple", ["ps_proba", "sr_proba"])
YEAR = 2011
SPAN = None

def get_word_proba(word, string):
    if word in string:
        return 1
    return 0

def get_class_word_proba(word, set_of_strings):
    total_documents = len(set_of_strings)
    sum_of_appareances = sum(map(lambda d: get_word_proba(word, d), set_of_strings))
    return sum_of_appareances/total_documents


with open("../CrossValidationFolds/fold_{}_span_{}.json".format(YEAR, SPAN), "r", encoding="utf-8") as fold_file:
    fold = json.load(fold_file)

#####CREATE DICTIONARY
print("creating dictionary")

words = {}  # indicates for each word (key), the number of SRs and PSs it appears in
all_fold_papers = fold["{}".format(YEAR)]["docs"] + fold["others"]["docs"]

for paper in all_fold_papers:

    check_occurence_sr = False
    check_occurence_ps = False
    for word in paper["abstract"]:
        if word not in words:
            words[word] = {"systematic-review": 0, "primary-study": 0}

        if paper["classification"] == "systematic-review" and not check_occurence_sr:
            words[word]["systematic-review"] += 1
            check_occurence_sr = True

        if paper["classification"] == "primary-study" and not check_occurence_ps:
            words[word]["primary-study"] += 1
            check_occurence_ps = True


print("dictionary done, words identified: ", len(words))
######CALCULATE PROBABILITIES PER CLASS
print("calculating probabilities")

results = {}

ps_total = sum((1 if paper["classification"] == "primary-study" else 0 for paper in all_fold_papers))
sr_total = sum((1 if paper["classification"] == "systematic-review" else 0 for paper in all_fold_papers))

i = 0
for word, occurrence_dict in words.items():
    ps_proba = occurrence_dict["primary-study"] / ps_total  # prob of the word appearing in a PS
    sr_proba = occurrence_dict["systematic-review"] / sr_total  # prob of the word appearing in a SR
    results[word] = {"ps_proba": ps_proba, "sr_proba": sr_proba}

    if i % 10000 == 0:
        print("{}/{}".format(i, len(words)))

    i += 1

######SAVE RESULTS

with open("word_appearances_prob.json", "w") as json_out:
    json.dump(results, json_out)
