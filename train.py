import argparse
import os
import pickle

import numpy
from sklearn.model_selection import train_test_split

from model import MorphologicalInflector
from constants import *

def load_training_data(file_name):
    """Loads training data.

    Args:
        file_name: path to file containing the data

    Returns:
        lemmas: list of lemma
        tags: list of tags
        inflected_forms: list of inflected form
    """

    with open(file_name) as file:
        text = file.read()

    lemmas = []
    tags = []
    inflected_forms = []

    for line in text.split('\n')[:-1]:
        lemma, inflected_form, tag = line.split('\t')
        lemmas.append(lemma)
        inflected_forms.append(inflected_form)
        tags.append(tag) 

    return lemmas, tags, inflected_forms

def get_index_dictionaries(lemmas, tags, inflected_forms):
    """Returns char2index, index2char, tag2index

    Args:
        lemmas: list of lemma
        tags: list of tags
        inflected_forms: list of inflected form

    Returns: 
        char2index: a dictionary which maps character to index
        index2char: a dictionary which maps indedx to character
        tag2index: a ditionary which maps morphological tag to index 
    """

    unique_chars = set(''.join(lemmas) + ''.join(inflected_forms))
    unique_chars.update(START_CHAR, STOP_CHAR) # special start and end symbols
    unique_chars.update(UNKNOWN_CHAR) # special charcter for unkown word
    char2index = {}
    index2char = {}

    for index, char in enumerate(unique_chars):
        char2index[char] = index
        index2char[index] = char

    unique_tags = set(';'.join(tags).split(';'))
    unique_tags.update(UNKNOWN_TAG)
    tag2index = {tag:index for index, tag in enumerate(unique_tags)}

    return char2index, index2char, tag2index

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('training_data', help='Path to training file.')
    parser.add_argument('--embedding_size', type=int, default=100, help='Embedding Size')
    parser.add_argument('--hidden_size', type=int, default=100, help='Hidden Size')
    parser.add_argument('--epochs', type=int, default=1, help='Epochs')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning Rate')
    args = parser.parse_args()

    lemmas, tags, inflected_forms = load_training_data(args.training_data)
    char2index, index2char, tag2index = get_index_dictionaries(lemmas, tags, inflected_forms)

    lemmas_train, lemmas_test, tags_train, tags_test, inflected_forms_train, inflected_forms_test = train_test_split(lemmas, tags, inflected_forms, train_size=0.8, test_size=0.2)

    model = MorphologicalInflector(char2index, index2char, tag2index, embedding_size=args.embedding_size, hidden_size=args.hidden_size)
    model.train(lemmas_train, tags_train, inflected_forms_train, epochs=args.epochs, learning_rate=args.learning_rate)
   
    # pickle.dump(model, 'model.pkl')
    # model = pickle.load('model.pkl')

    print("Correct\tPredicted")
    for correct, predicted in zip(inflected_forms_test, model.generate(lemmas_test, tags_test)):
        print("{}\t{}".format(correct, predicted))