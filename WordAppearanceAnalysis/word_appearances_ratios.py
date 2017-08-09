import json
import matplotlib.pyplot as plt
from numpy import log

with open("word_appearances_prob.json", "r", encoding="utf-8") as json_in:
    data = json.load(json_in)

numerator = "ps_proba"
denominator = "sr_proba"

proba_ratios = {}
for word, probabilities in data.items():
    proba_ratios[word] = (probabilities[numerator] + 1) / (probabilities[denominator] + len(data))


results = sorted(proba_ratios.items(), key=lambda t: t[1], reverse=True)  # sort ratios from greatest to smallest

# plt.ylabel("{}/{}".format(denominator, numerator))
# plt.xlabel("Orden de palabras con mayor ratio")
# for i in range(len(results)):
#     plt.scatter(log(i), log(results[i][1]))
#
# plt.savefig("proba_ratio_results_log.png")

for word in results[:20]:
    print(word[0], data[word[0]][numerator])
print("\n")
for word in results[-20:]:
    print(word[0], data[word[0]][denominator])

with open("word_appearances_ratios.json", "w") as json_out:
    json.dump(results, json_out)
