from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

import numpy as np
import argparse
import pickle
import json
import os


class DocumentSpace:

    def __init__(self, language_model=None, lang_mod_order=None, span=None):
        self.span = span
        self.lang_mod_order = lang_mod_order  # list of words of the language model
        self.language_model = language_model

    def get_abs_vectors(self, papers):

        """ Generates a vector for every abstract to be classified.
        :parameter papers: JSON array (python list) containing
        the processed papers, represented as dicts."""

        labels = [paper["classification"] for paper in papers]  # ground truth

        print("Calculating abstracts' vectors...")
        parsed = 0
        n_abstracts = len(papers)
        print("started vector build")
        pooled_vectors = list()
        hash_table = {word: index for (index, word) in enumerate(self.lang_mod_order)}

        for paper in papers:

            word_mtx = list()  # vectors for each word of the document, up to span (if span is specified)
            words = paper["abstract"]  # This assumes the abstract is already divided into words
            word_count = 0
            # loop through words and see if they are in the vocab, so that they can be assigned a vector
            while (self.span and word_count < self.span) or word_count < len(words):
                word = words[word_count]
                word_mtx.append(self.language_model[hash_table[word] if word in hash_table else 0])
                word_count += 1
            try:
                pooled_vector = np.amax(word_mtx, axis=0)
            except ValueError:  # no words
                pooled_vector = self.language_model[0]  # unk

            pooled_vectors.append(pooled_vector)

            if not parsed % 10000:
                print("--->{}/{}".format(parsed, n_abstracts), end="\r")
            parsed += 1
        print("--->{}/{}".format(parsed, n_abstracts))
        assert n_abstracts == parsed
        print("Finished calculating abstracts' vectors.")
        return labels, pooled_vectors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify papers.")
    parser.add_argument("--K", type=int, required=True, help="Number of nearest neighbours to consider")
    parser.add_argument("--model_path", help="Path to a folder containing everything related to the model, namely "
                                             "embeddings and vocab files.",
                        required=True)
    parser.add_argument("--span", help="Number of words used for classification, counting from "
                                       "the start of the abstract", type=int, default=10)
    parser.add_argument("--KNN_papers_set", help="Path to the KNN paper set.",
                        required=True)
    parser.add_argument("--distance_metric", default="euclidean", help="Metric to use to select nearest neighbours. "
                                                                       "Currently euclidean and dot product are "
                                                                       "implemented.")
    parser.add_argument("--algorithm", default="ball_tree", help="Algorithm used to compute KNN-search as described in "
                                                                 "Scikit-Learn documentation.")

    args = parser.parse_args()

    path_to_model = args.model_path
    path_to_papers = args.KNN_papers_set
    join_path = os.path.join

    with open(join_path(path_to_model, "embeddings"), 'rb') as embeddings, \
            open(join_path(path_to_model, "vocab.txt")) as vocab, \
            open(join_path(path_to_papers, "train_papers")) as train_set, \
            open(join_path(path_to_papers, "test_papers")) as test_set:

        model = pickle.load(embeddings)
        model_order = [line.split()[0][2: -1] for line in vocab]

        train = json.load(train_set)
        test = json.load(test_set)

    Space = DocumentSpace(model, model_order, args.span)
    train_labels, train_data = Space.get_abs_vectors(train)
    test_labels, test_data = Space.get_abs_vectors(test)

    if args.distance_metric == "dot":
        args.distance_metric = np.dot

    classifier = KNeighborsClassifier(n_neighbors=args.K, metric=args.distance_metric,
                                      algorithm=args.algorithm, n_jobs=-1)
    print("fitting")
    classifier.fit(train_data, train_labels)
    print("predicting")

    prediction = classifier.predict(test_data)  # in the same order as the testing documents, so it CAN be known how each document was classified

    classes = ["primary-study", "systematic-review"]

    if len(test_labels) != len(prediction):
        print("dimensions error. labels: {}, predictions: {}".format(len(test_labels), len(prediction)))

    class_dimension = len(classes)
    conf_mtx = np.zeros([class_dimension, class_dimension])
    for i in range(0, len(prediction)):
        predicted_class = classes.index(prediction[i])
        actual_class = classes.index(test_labels[i])
        conf_mtx[actual_class][predicted_class] += 1
    np.set_printoptions(suppress=True)
    print(conf_mtx)
    print('')
    accuracy = (sum(conf_mtx[i][i] for i in range(0, len(classes)))/len(prediction))
    print('Accuracy: {}'.format(accuracy))

    #Uncomment below for more specific metrics
    recall = lambda i: (conf_mtx[i][i]/sum(conf_mtx[i][j] for j in range(0,class_dimension)))
    recall_sum = 0
    recall_list = []
    for i in range(0,class_dimension):
        rcl = recall(i)
        if not np.isnan(rcl):
            recall_sum += rcl
        recall_list.append((i, rcl))
        print('Recall {}: {:.5f}'.format(i, rcl))
    print()
    recall_mean = recall_sum/class_dimension
    print('Recall macro average: {:.5f}'.format(recall_mean))
    micro_recall = recall_score(test_labels, prediction, average='weighted')
    print('Recall weighted average: {:.5f}'.format(micro_recall))

    precision = lambda i: (conf_mtx[i][i]/sum(conf_mtx[j][i] for j in range(0, class_dimension)))
    precision_sum = 0
    precision_list = list()
    for i in range(0,class_dimension):
        label_precision = precision(i)
        if not np.isnan(label_precision):
            precision_sum += label_precision
        precision_list.append((i, label_precision))
        print('Precision {}: {:.5f}'.format(i, label_precision))
    print()
    precision_mean = precision_sum/class_dimension
    print('Precision macro average: {:.5f}'.format(precision_mean))
    micro_precision = precision_score(test_labels, prediction, average='weighted')
    print('Precision weighted average: {:.5f}'.format(micro_precision))

    f1 = f1_score(test_labels, prediction, average='weighted')
    print('F1 score weighted average: {:.5f}'.format(f1))

    output = ''
    output += 'Model: {}\n'.format(args.model_path)
    output += 'KNN classifier with k = {}\n'.format(args.K)
    output += 'span = {}\n'.format(args.span)
    output += 'Set: {}\n'.format(args.KNN_papers_set)
    output += 'Accuracy : {}\n'.format(accuracy)
    output += "RECALL\n"
    for rcl in recall_list:
        output += 'Recall {}: {:.5f}\n'.format(rcl[0], rcl[1])
    output += 'Recall mean: {:.5f}\n'.format(recall_mean)
    output += 'Recall weighted average: {:.5f}\n'.format(micro_recall)
    output += "PRECISION\n"
    for pcsn in precision_list:
        output += 'Precision {}: {:.5f}\n'.format(pcsn[0], pcsn[1])
    output += 'Precision mean: {:.5f}\n'.format(precision_mean)
    output += 'Precision weighted average: {:.5f}\n'.format(micro_precision)
    output += 'F1 score weighted average: {:.5f}\n'.format(f1)
    output += 'CONFUSSION MATRIX\n'
    output += str(conf_mtx)
    #output += Space.max_pool_lab.infographic_from_results()

    #Save results for later analysis:
    with open("test_output_no_exclusives_{}.txt".format("lol"), "w") as out_file: #add something to distinguish
        out_file.write(output)

    # return accuracy, conf_mtx  # , restricted_dict

    # '''with open("Max_pool_lab_results{}".format(save_id), "wb") as out_mpl:
    #     pickle.dump(Space.max_pool_lab.obtain_results(), out_mpl)'''
    #
    # '''with open("predictions_proba{}".format(save_id), "wb") as out_pp:
    #     pickle.dump(predictions, out_pp)
    #
    # with open("labels{}".format(save_id), "wb") as out_lbl:
    #     pickle.dump(test_labels, out_lbl)'''
