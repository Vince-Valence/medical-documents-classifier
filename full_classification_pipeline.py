import string
import json
import pickle
import os
import io
import sklearn.metrics as metrics
import tabulate
import csv
import winsound
from datetime import datetime
from classify import DocumentSpace
from sklearn.neighbors import KNeighborsClassifier
from collections import OrderedDict
from random import sample

YEARS = (2011,)
MODELS = (9,)
WORDS = (None,)  # none means all
KNN_Ks = (10,)
MODEL_VALIDATION = True  # whether to validate the models or not. If false, the 2 params below are ignored
MODEL_VALIDATION_WORDS = (100000, 165000, 5, 10, 25, 50, 100, 250, 500, 2500, 5000, 25000, 50000)  # words for ratio and random select
MODEL_VALIDATION_ONLY = True  # if true, no regular classification will be made

start_time = datetime.now()

log = open("full_pipeline_log.txt", 'a')  # save runtime information here
log.write("Initiated execution of full_classification_pipeline.py on {} with parameters:\n".format(start_time))
print("Initiated execution of full_classification_pipeline.py on {} with parameters:\n".format(start_time))
log.write("{} = {}\n{} = {}\n{} = {}\n{} = {}\n{} = {}\n{} = {}\n{} = {}\n".format("YEARS", YEARS, "MODELS",MODELS, "WORDS", WORDS, "KNN_Ks", KNN_Ks, "MODEL_VALIDATION", MODEL_VALIDATION, "MODEL_VALIDATION_WORDS", MODEL_VALIDATION_WORDS, "MODEL_VALIDATION_ONLY", MODEL_VALIDATION_ONLY))
print("{} = {}\n{} = {}\n{} = {}\n{} = {}\n{} = {}\n{} = {}\n{} = {}\n".format("YEARS", YEARS, "MODELS",MODELS, "WORDS", WORDS, "KNN_Ks", KNN_Ks, "MODEL_VALIDATION", MODEL_VALIDATION, "MODEL_VALIDATION_WORDS", MODEL_VALIDATION_WORDS, "MODEL_VALIDATION_ONLY", MODEL_VALIDATION_ONLY))

path_to_docs = "../../../PaperProcessing/documents_array.json"
path_to_embeddings_folder = "../../../LanguageModels/SkipGram/Advanced/TrainedModels/Model{}"


# open original documents' file:
with open(path_to_docs, encoding="utf-8") as docs_json:
    log.write("Loading original documents...\n")
    print("Loading original documents...\n")
    load_docs_start = datetime.now()
    documents = json.load(docs_json)
    load_docs_time = datetime.now() - load_docs_start
    log.write("Loaded {} documents in {}.\n".format(len(documents), load_docs_time))
    print("Loaded {} documents in {}.\n".format(len(documents), load_docs_time))


# clean abstracts:
def parse_abstract(abstract):
    abstract = abstract.lower()
    words = abstract.split()
    clean = [parse_word(word) for word in words]
    return list(filter(lambda l: l, clean))

keep = string.ascii_lowercase + '-'
dismiss = "0123456789"


def parse_word(word):
    for ch in dismiss:
        if ch in word:
            return None
    return "".join(ch for ch in word if ch in keep)


log.write("Parsing abstracts...\n")
print("Parsing abstracts...\n")
parse_abstracts_start = datetime.now()
for doc in documents:
    abstract = doc["abstract"]
    if abstract:
        doc["abstract"] = parse_abstract(abstract)
log.write("Abstracts parsed in {}.\n".format(datetime.now() - parse_abstracts_start))
print("Abstracts parsed in {}.\n".format(datetime.now() - parse_abstracts_start))

# The docs now have their abstracts parsed


# work only with SRs and PSs:
log.write("Filtering documents: only considering systematic reviews and primary studies...\n")
print("Filtering documents: only considering systematic reviews and primary studies...\n")
sr_ps_filter_start = datetime.now()
docs_sr_ps = [doc for doc in filter(lambda d: d["classification"] in ("primary-study", "systematic-review"), documents)]
log.write("Filtered in {}.\n".format(datetime.now() - sr_ps_filter_start))
print("Filtered in {}.\n".format(datetime.now() - sr_ps_filter_start))

doc_space = DocumentSpace()

log.write("Engaging classification loop...\n")
print("Engaging classification loop...\n")
for w2v_model in MODELS:
    current_embeddings_folder = path_to_embeddings_folder.format(w2v_model)

    with open(os.path.join(current_embeddings_folder, "embeddings_normalized"), 'rb') as embeddings_file, \
            open(os.path.join(current_embeddings_folder, "vocab.txt")) as vocab_file:
        embeddings = pickle.load(embeddings_file)
        vocab = [line.split()[0][2: -1] for line in vocab_file]
        vocab_hash = {word: index for (index, word) in enumerate(vocab)}
        all_vocabs = [] if MODEL_VALIDATION_ONLY and MODEL_VALIDATION else [vocab]
        all_vocabs_types = [] if MODEL_VALIDATION_ONLY and MODEL_VALIDATION else ["normal"]
        all_vocabs_true_words = [] if MODEL_VALIDATION and MODEL_VALIDATION_ONLY else [len(vocab) - 1]

    doc_space.language_model = embeddings

    if MODEL_VALIDATION:
        log.write("Creating vocabulary variants for model validation...\n")
        print("Creating vocabulary variants for model validation...\n")
        vocab_variants_start = datetime.now()
        # create vocabulary variants
        with open("Word_Proba/word_appearances_ratios.json", encoding="utf-8") as ratios_file:
            ratios = json.load(ratios_file)

        new_vocabs = []
        for n_words in MODEL_VALIDATION_WORDS:
            log.write("Creating vocabulary variants with {} words (at most)...\n".format(n_words))
            print("Creating vocabulary variants with {} words (at most)...\n".format(n_words))
            current_vocabs_start = datetime.now()
            # Get at most n_words with high ratios
            high_ratio_words = []
            i = 0
            while len(high_ratio_words) < n_words and i < len(ratios)//2:
                # if ratios[i][0] in vocab_hash:
                high_ratio_words.append(ratios[i][0])
                i += 1
            # Get at most n_words with low ratios
            low_ratio_words = []
            i = len(ratios) - 1
            while len(low_ratio_words) < n_words and i > len(ratios)//2:
                # if ratios[i][0] in vocab_hash:
                low_ratio_words.append(ratios[i][0])
                i -= 1
            print("Using {} high ratio words. Aimed for {}\n".format(len(high_ratio_words), n_words))
            log.write("Using {} high ratio words. Aimed for {}\n".format(len(high_ratio_words), n_words))
            print("Using {} low ratio words. Aimed for {}\n".format(len(low_ratio_words), n_words))
            log.write("Using {} low ratio words. Aimed for {}\n".format(len(low_ratio_words), n_words))
            # Create a new vocab replacing the words that are not in the high and low ratios words with FORBIDDEN
            # except for UNK
            high_and_low_hash = dict.fromkeys(high_ratio_words + low_ratio_words)
            new_vocab = [vocab[0]] + [word if word in high_and_low_hash else "FORBIDDEN" for word in vocab[1:]]
            all_vocabs.append(new_vocab)
            all_vocabs_true_words.append(len(high_and_low_hash))

            # Get at most 2n_words (the same number of high and low ratios words) at random
            ratios_words = [ratio[0] for ratio in ratios]
            smp_hash = dict.fromkeys(sample(ratios_words, len(high_and_low_hash)))
            new_vocab = [vocab[0]] + [word if word in smp_hash else "FORBIDDEN" for word in vocab[1:]]
            all_vocabs.append(new_vocab)
            all_vocabs_true_words.append(len(high_and_low_hash))
            all_vocabs_types.extend(("ratio select", "random select"))
            print("Finished vocabs variants with {} words (at most) in {}\n".format(n_words, datetime.now() - current_vocabs_start))
            log.write("Finished vocabs variants with {} words (at most) in {}\n".format(n_words, datetime.now() - current_vocabs_start))
        print("Finished calculating vocab variants for model {} in {}\n".format(w2v_model, datetime.now() - vocab_variants_start))
        log.write("Finished calculating vocab variants for model {} in {}\n".format(w2v_model, datetime.now() - vocab_variants_start))

    for vocab, vocab_type, vocab_true_words in zip(all_vocabs, all_vocabs_types, all_vocabs_true_words):
        doc_space.lang_mod_order = vocab

        # 10-fold cross validation's folds by years (2002 to 2011):
        for year in YEARS:

            # filter by span
            for span in WORDS:

                if not os.path.isfile("fold_{}_span_{}.json".format(year, span)):  # if the fold doesn't exist as a file

                    year_docs = [d for d in filter(lambda doc: str(year) in str(doc["year"]) and doc["abstract"], docs_sr_ps)]
                    other_docs = [d for d in filter(lambda doc: str(year) not in str(doc["year"]) and doc["abstract"], docs_sr_ps)]

                    if span:
                        year_docs = [d for d in filter(lambda doc: len(doc["abstract"]) >= span, year_docs)]
                        other_docs = [d for d in filter(lambda doc: len(doc["abstract"]) >= span, other_docs)]

                    with io.open("fold_{}_span_{}.json".format(year, span), "w", encoding="utf-8") as fold_file:
                        json.dump({"{}".format(year): {"count": len(year_docs), "docs": year_docs},
                                   "others": {"count": len(other_docs), "docs": other_docs}}, fold_file, ensure_ascii=False)
                else:  # if it exists as a file
                    with open("fold_{}_span_{}.json".format(year, span), encoding="utf-8") as fold_file:
                        fold_docs = json.load(fold_file)
                        year_docs = fold_docs["{}".format(year)]["docs"]
                        other_docs = fold_docs["others"]["docs"]

                if span:
                    doc_space.span = span

                log.write("Building vectors for each abstract...\n")
                print("Building vectors for each abstract...\n")
                build_vectors_start = datetime.now()
                train_labels, train_vectors = doc_space.get_abs_vectors(other_docs)
                test_labels, test_vectors = doc_space.get_abs_vectors(year_docs)
                log.write("Finished building vectors in {}.\n".format(datetime.now() - build_vectors_start))
                print("Finished building vectors in {}.\n".format(datetime.now() - build_vectors_start))

                for K in KNN_Ks:

                    classifier = KNeighborsClassifier(n_neighbors=K, metric="euclidean", algorithm="ball_tree", n_jobs=-1)
                    classifier.fit(train_vectors, train_labels)
                    log.write("Predicting with:\nModel = {}\nYear = {}\nSpan = {}\nK = {}\n...\n".format(w2v_model, year, span, K))
                    print("Predicting with:\nModel = {}\nYear = {}\nSpan = {}\nK = {}\n...\n".format(w2v_model, year, span, K))
                    prediction_start = datetime.now()
                    prediction = classifier.predict(test_vectors)
                    log.write("Finished predicting in {}.\n".format(datetime.now() - prediction_start))
                    print("Finished predicting in {}.\n".format(datetime.now() - prediction_start))

                    log.write("Calculating metrics...\n")
                    print("Calculating metrics...\n")
                    accuracy = metrics.accuracy_score(test_labels, prediction)
                    log.write("Accuracy: {}\n".format(accuracy))
                    print("Accuracy: {}\n".format(accuracy))

                    precision_ps, precision_sr = metrics.precision_score(test_labels, prediction, average=None, labels=["primary-study", "systematic-review"])
                    log.write("Precision PS: {}\nPrecision SR: {}\n".format(precision_ps, precision_sr))
                    print("Precision PS: {}\nPrecision SR: {}\n".format(precision_ps, precision_sr))

                    recall_ps, recall_sr = metrics.recall_score(test_labels, prediction, average=None, labels=["primary-study", "systematic-review"])
                    log.write("Recall PS: {}\nRecall SR: {}\n".format(recall_ps, recall_sr))
                    print("Recall PS: {}\nRecall SR: {}\n".format(recall_ps, recall_sr))

                    f1_ps, f1_sr = metrics.f1_score(test_labels, prediction, average=None, labels=["primary-study", "systematic-review"])
                    log.write("F1 PS: {}\nF1 SR: {}\n".format(f1_ps, f1_sr))
                    print("F1 PS: {}\nF1 SR: {}\n".format(f1_ps, f1_sr))

                    conf_mtx = metrics.confusion_matrix(test_labels, prediction, labels=["primary-study", "systematic-review"])
                    log.write("Confusion matrix:\n\n")
                    print("Confusion matrix:\n\n")
                    tab_conf_mtx = tabulate.tabulate(conf_mtx, headers=["primary-study", "systematic-review"],
                                                     showindex=["primary-study", "systematic-review"])
                    log.write(tab_conf_mtx + "\n\n")
                    print(tab_conf_mtx + "\n\n")

                    log.write("Saving results...\n")
                    print("Saving results...\n")

                    results = OrderedDict([("accuracy", accuracy), ("precision_ps", precision_ps),
                                ("precision_sr", precision_sr), ("recall_ps", recall_ps), ("recall_sr", recall_sr), ("f1_ps", f1_ps),
                                ("f1_sr", f1_sr), ("confusion_matrix", tab_conf_mtx), ("prediction", prediction.tolist())])

                    if vocab_type == "ratio select":
                        results["ratio_select"] = vocab_true_words
                        results["vocab"] = vocab
                    elif vocab_type == "random select":
                        results["random_select"] = vocab_true_words
                        results["vocab"] = vocab

                    with open("prediction_model_{}_year_{}_span_{}_{}.json".format(w2v_model, year, span, "{}_{}".format(vocab_type, vocab_true_words) if vocab_type != "normal" else ""), "w",
                              encoding="utf-8") as prediction_file:
                        json.dump(results, prediction_file)

                    exists = os.path.isfile("all_results.csv")
                    with open("all_results.csv", "a", newline="") as results_file:

                        headers = ["Model", "Span", "K", "Test Year", "Accuracy", "Recall PS", "Recall SR", "Precision PS", "Precision SR", "F1 PS", "F1 SR", "True PS", "True SR", "False PS", "False SR", "Especial Select", "Especial Select Words"]
                        writer = csv.DictWriter(results_file, headers)
                        if not exists:
                            writer.writeheader()
                        writer.writerow({"Model": w2v_model, "Span": span, "K": K, "Test Year": year, "Accuracy": accuracy,
                                         "Recall PS": recall_ps, "Recall SR": recall_sr, "Precision PS": precision_ps,
                                         "Precision SR": precision_sr, "F1 PS": f1_ps, "F1 SR": f1_sr, "True PS": conf_mtx[0][0],
                                         "True SR": conf_mtx[1][1], "False PS": conf_mtx[1][0], "False SR": conf_mtx[0][1],
                                         "Especial Select": vocab_type, "Especial Select Words": vocab_true_words})

                    log.write("Results saved.\n")
                    print("Results saved.\n")


finish_time = datetime.now()
elapsed = finish_time - start_time
log.write("Finished execution of full_classification_pipeline.py on {}. Time elapsed: {}\n\n".format(finish_time, elapsed))
print("Finished execution of full_classification_pipeline.py on {}. Time elapsed: {}\n\n".format(finish_time, elapsed))
log.close()  # close log file
winsound.Beep(300, 2000)
