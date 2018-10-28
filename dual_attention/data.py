import os
import random

import torch
from sklearn.model_selection import train_test_split

from constants import TASK1_DATA_PATH
from data import read_dataset, read_covered_dataset
from m2maligner import one_one_alignment
from utils import accuracy, average_distance, grouper

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Vocab:
    def __init__(self, lemmas, tags, inflected_forms):
        self.START_CHAR = '⏵'
        self.STOP_CHAR = '⏹'
        self.UNKNOWN_CHAR = '⊗'
        self.UNKNOWN_TAG = '⊤'
        self.PAD_CHAR = '₮'

        self.build_char_dicts(lemmas, inflected_forms)
        self.build_tag_dict(tags)

        self.padding_idx = self.char_to_index(self.PAD_CHAR)

    def build_char_dicts(self, lemmas, inflected_forms):

        unique_chars = set(''.join(lemmas) + ''.join(inflected_forms))
        unique_chars.update(self.START_CHAR, self.STOP_CHAR)  # special start and end symbols
        unique_chars.update(self.UNKNOWN_CHAR)  # special character for unknown word
        self.char2index = {}
        self.index2char = {}

        self.char2index[self.PAD_CHAR] = 0
        self.index2char[0] = self.PAD_CHAR

        for index, char in enumerate(unique_chars):
            self.char2index[char] = index + 1
            self.index2char[index + 1] = char

        self.char_vocab_size = len(self.index2char.keys())

    def build_tag_dict(self, tags):
        unique_tags = set(';'.join(tags).split(';'))
        unique_tags.update(self.UNKNOWN_TAG)
        self.tag2index = {tag: index + 1 for index, tag in enumerate(unique_tags)}
        self.tag2index[self.PAD_CHAR] = 0
        self.tag_vocab_size = len(self.tag2index.keys())

    def char_to_index(self, char):
        return self.char2index.get(char, self.char2index[self.UNKNOWN_CHAR])

    def index_to_char(self, index):
        return self.index2char.get(index, self.UNKNOWN_CHAR)

    def tag_to_index(self, tag):
        return self.tag2index.get(tag, self.tag2index[self.UNKNOWN_TAG])

    def words_to_indices(self, words, tensor=False, start_char=False, stop_char=False):
        """Converts list of words to a list with list containing indices

        Args:
            words: list of words
            tensor: if to return a list of tensor
            start_char: whether to add start character
            stop_char: whether to add stop character

        Returns:
            tensor: list of list/tensor containing indices for a sequence of characters
        """

        list_indices = []
        for word in words:
            word_indices = [self.char_to_index(char) for char in word]

            if start_char:
                word_indices = [self.char_to_index(self.START_CHAR)] + word_indices

            if stop_char:
                word_indices.append(self.char_to_index(self.STOP_CHAR))

            if tensor:
                word_indices = torch.Tensor(word_indices).to(device)

            list_indices.append(word_indices)

        return list_indices

    def tag_to_indices(self, tags):
        """Converts list of tags to a list of lists containing indices

        Args:
            tags: list of tags

        Returns:
            tensor: list of list containing indices of sub_tags
        """

        list_indices = [[self.tag_to_index(sub_tag) for sub_tag in tag.split(';')] for tag in tags]
        return list_indices

    def indices_to_word(self, indices):
        """Returns a word given lists containing indices of words.

        Args:
            indices: lists containing indices

        Returns:
            word: list of strings
        """

        return [''.join([self.index_to_char(int(index)) for index in indices_seq]) for indices_seq in indices]


def get_p_gens(srcs, tgts, alignments):
    """Returns target p_gens."""
    p_gen = []
    for src, tgt, alignment in zip(srcs, tgts, alignments):
        seq_p_gen = []
        for i, index in enumerate(alignment):
            if index == -1 or src[index] != tgt[i]:
                seq_p_gen.append(1)
            else:
                seq_p_gen.append(0)
        p_gen.append(seq_p_gen)
    return p_gen


def get_alignment(lemmas, inflected_forms, vocab):
    alignments = one_one_alignment([list(lemma) + [vocab.STOP_CHAR] for lemma in lemmas],
                                   [list(inflected_form) + [vocab.STOP_CHAR] for inflected_form in inflected_forms])
    for alignment_seq in alignments:
        for i in range(len(alignment_seq)):
                if alignment_seq[i] != -1:
                    alignment_seq[i] += 1
    return alignments


def load_data(language, dataset, test_data='dev', val_ratio=0.2, random_state=42, use_external_val_data=False):
    """Loads training data."""

    train_dataset = os.path.join(TASK1_DATA_PATH, '{}-train-{}'.format(language, dataset))
    lemmas, tags, inflected_forms = read_dataset(train_dataset)
    train_data_size = len(lemmas)

    if val_ratio*train_data_size > 1000:
        val_ratio = 1000/train_data_size
    val_dataset = None

    if use_external_val_data:
        dev_dataset = os.path.join(TASK1_DATA_PATH, '{}-dev'.format(language))
        high_dataset = os.path.join(TASK1_DATA_PATH, '{}-train-high'.format(language))
        medium_dataset = os.path.join(TASK1_DATA_PATH, '{}-train-medium'.format(language))
        low_dataset = os.path.join(TASK1_DATA_PATH, '{}-train-low'.format(language))

        if test_data != 'dev':
            val_dataset = dev_dataset
        elif os.path.exists(high_dataset) and train_dataset != high_dataset:
            val_dataset = high_dataset
        elif os.path.exists(medium_dataset) and train_dataset != medium_dataset:
            val_dataset = medium_dataset
        elif os.path.exists(low_dataset) and train_dataset != low_dataset:
            val_dataset = low_dataset

        if val_dataset is not None:
            lemmas_val, tags_val, inflected_forms_val = read_dataset(val_dataset)

    if val_dataset is not None and len(lemmas_val) >= val_ratio*train_data_size:
        lemmas_train, tags_train, inflected_forms_train = lemmas, tags, inflected_forms

        val_data = list(zip(lemmas_val, tags_val, inflected_forms_val))
        random.seed(random_state)
        val_data_size = int(min(max(val_ratio*train_data_size, 100), len(lemmas_val)))
        val_data = random.sample(val_data, val_data_size)
        lemmas_val, tags_val, inflected_forms_val = zip(*val_data)
        lemmas_val, tags_val, inflected_forms_val = list(lemmas_val), list(tags_val), list(inflected_forms_val)
    else:
        lemmas_train, lemmas_val, tags_train, tags_val, inflected_forms_train, inflected_forms_val = train_test_split(
            lemmas, tags, inflected_forms, test_size=val_ratio, random_state=random_state)

    train_data_size = len(lemmas_train)
    val_data_size = len(lemmas_val)

    if test_data == 'dev':
        dev_data = os.path.join(TASK1_DATA_PATH, '{}-dev'.format(language))
        lemmas_test, tags_test, _ = read_dataset(dev_data)
    elif test_data == 'test':
        test_data = os.path.join(TASK1_DATA_PATH, '{}-covered-test'.format(language))
        lemmas_test, tags_test = read_covered_dataset(test_data)
    else:
        lemmas_test, tags_test, inflected_forms_test = [], [], []

    vocab = Vocab(lemmas_train+lemmas_val+lemmas_test, tags_train+tags_val+tags_test, inflected_forms_train)

    alignments_train = get_alignment(lemmas_train, inflected_forms_train, vocab)
    p_gens_train = get_p_gens([[vocab.START_CHAR] + list(lemma) + [vocab.STOP_CHAR] for lemma in lemmas_train],
                              [list(inflected_form) + [vocab.STOP_CHAR] for inflected_form in inflected_forms_train], alignments_train)

    alignments_val = get_alignment(lemmas_train+lemmas_val, inflected_forms_train+inflected_forms_val, vocab)[train_data_size:]
    p_gens_val = get_p_gens([[vocab.START_CHAR] + list(lemma) + [vocab.STOP_CHAR] for lemma in lemmas_val],
                              [list(inflected_form) + [vocab.STOP_CHAR] for inflected_form in inflected_forms_val], alignments_val)

    lemmas_indices = vocab.words_to_indices(lemmas_train+lemmas_val, start_char=True, stop_char=True)
    tags_indices = vocab.tag_to_indices(tags_train+tags_val)
    inflected_forms_indices = vocab.words_to_indices(inflected_forms_train+inflected_forms_val)

    model_inputs_train = list(zip(lemmas_indices[:train_data_size], tags_indices[:train_data_size]))
    labels_train = list(zip(inflected_forms_indices[:train_data_size], alignments_train, p_gens_train))

    model_inputs_val = list(zip(lemmas_indices[train_data_size:], tags_indices[train_data_size:]))
    labels_val = list(zip(inflected_forms_indices[train_data_size:], alignments_val, p_gens_val))

    return model_inputs_train, model_inputs_val, labels_train, labels_val, vocab


def evaluate_on_dev(model, filename, batch_size=32):
    """Prints predictions and metrics by model on development dataset."""

    lemmas, tags, inflected_forms = read_dataset(filename)
    predictions = generate_predictions(model, lemmas, tags, batch_size)

    for prediction in predictions:
        print(prediction)

    print()
    print("Accuracy: {}, Average Distance: {}".format(accuracy(predictions, inflected_forms), average_distance(predictions, inflected_forms)))


def generate_predictions(model, lemmas, tags, batch_size=32):
    """Returns predicted inflected forms for given lemmas and tags."""

    lemmas_indices = model.vocab.words_to_indices(lemmas, start_char=True, stop_char=True)
    tags_indices = model.vocab.tag_to_indices(tags)
    model_input = list(zip(lemmas_indices, tags_indices))

    predictions = []

    for batch_input in grouper(model_input, batch_size):
        batch_input = list(filter(lambda x: x is not None, batch_input))  # remove None objects introduced by grouper

        # set model to evaluating mode
        model.eval()

        # compute model output and loss
        p_ws, a_ls, p_gens = model(*zip(*batch_input))
        batch_predictions = [word.split(model.vocab.STOP_CHAR)[0] for word in model.vocab.indices_to_word(p_ws.argmax(2))]
        predictions += batch_predictions

    return predictions
