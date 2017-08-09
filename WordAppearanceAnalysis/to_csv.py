import json
import csv

with open("word_appearances_prob.json", encoding="utf-8") as prob_file, \
    open("word_appearances_ratios.json", encoding="utf-8") as ratio_file:
    probabilities = json.load(prob_file)
    ratios = json.load(ratio_file)


headers = ["Word", "Prob PS", "Prob SR", "Ratio (Prob PS/Prob RS)"]
with open("ratios_probs.csv", "w", newline="") as results_file:
    writer = csv.DictWriter(results_file, headers)
    writer.writeheader()
    for word, ratio in ratios:
        writer.writerow(dict(zip(headers, (word, probabilities[word]["ps_proba"], probabilities[word]["sr_proba"], ratio))))