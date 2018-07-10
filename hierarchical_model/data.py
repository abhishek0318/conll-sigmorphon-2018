import torch

from data import read_dataset
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


def load_data(filename):
    """Loads training data."""
    lemmas, tags, inflected_forms = read_dataset(filename)
    vocab = Vocab(lemmas, tags, inflected_forms)
    alignments = one_one_alignment([list(lemma) for lemma in lemmas], [list(inflected_form) for inflected_form in inflected_forms])
    p_gens = get_p_gens([list(lemma) for lemma in lemmas], [list(inflected_form) for inflected_form in inflected_forms], alignments)

    lemmas_indices = vocab.words_to_indices(lemmas, start_char=True, stop_char=True)
    tags_indices = vocab.tag_to_indices(tags)
    inflected_forms_indices = vocab.words_to_indices(inflected_forms)

    model_inputs = list(zip(lemmas_indices, tags_indices))
    labels = list(zip(inflected_forms_indices, alignments, p_gens))

    return model_inputs, labels, vocab


def evaluate_on_dev(model, filename, batch_size=32):
    """Prints predictions and metrics by model on development dataset."""

    lemmas, tags, inflected_forms = read_dataset(filename)
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

    for lemma, tag, inflected_form, prediction in zip(lemmas, tags, inflected_forms, predictions):
        print('{}\t{}\t{}\t{}'.format(lemma, tag, inflected_form, prediction))
    print(average_distance(predictions, inflected_forms))
    print(accuracy(predictions, inflected_forms))