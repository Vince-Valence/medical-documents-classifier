import tensorflow as tf
import os
import pickle

with tf.Session().as_default():

    model_folder = "Model{}"

    for model in range(7, 17):
        print("Current model: {}...\n".format(model))
        with open(os.path.join(model_folder.format(model), "embeddings"), 'rb') as embeddings_file:
            with tf.device('/cpu:0'):
                normalized_embeddings = tf.nn.l2_normalize(pickle.load(embeddings_file), 1).eval()
        with open(os.path.join(model_folder.format(model), "embeddings_normalized"), 'wb') as norm_emb_file:
            pickle.dump(normalized_embeddings, norm_emb_file)