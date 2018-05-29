import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from tqdm import tqdm, trange

from constants import *

class EncoderDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, additional_features_length, start_index, stop_index, max_output_length):
        """Initialised the network

        Args:
            vocab_size: size of input/output vacabulary.
            embedding_size: size of embeddings.
            hidden_size: size of hidden layer in encoder/decoder.
            additional_features_length: length of additional features to be used while decoding.
            start_index: index passed at first timestep to decoder
            stop_index: index on whose appearance the decoder stops decoding.
            max_output_length: maximum length of output

        Returns:
            nothing
        """

        super(EncoderDecoder, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embedding_layer = nn.Embedding(vocab_size, embedding_size)
        self.encoder = nn.LSTM(embedding_size, hidden_size)
        self.decoder = nn.LSTM(additional_features_length + embedding_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=2)
        self.start_index = start_index
        self.stop_index = stop_index
        self.max_output_length = max_output_length
 
    def forward(self, input_sequence, additional_features, decoder_inputs=None):
        """Function run in forward pass.

        Args:
            input_sequence: input_sequence (tensor) which is encoded
            additional_features: features other than input passed to input at each timestep while decoding.
            decoder_inputs: optional input to decoder while decoding (previous timestep ground truth label - teacher forcing)
                            if none, then the predicted output from the previous step is used.
        """

        embeddings = self.embedding_layer(input_sequence)
        _, (h_n, c_n) = self.encoder(embeddings.view(-1, 1, self.embedding_size))

        h_t, c_t = h_n, c_n # last hidden state of encoder, the initial state of decoder

        outputs = []

        if decoder_inputs is not None:
            for decoder_input in decoder_inputs:
                decoder_input_embedding = self.embedding_layer(decoder_input)
                output, (h_t, c_t) = self.decoder(torch.cat([decoder_input_embedding.view(-1), additional_features]).view(1, 1, -1), (h_t, c_t))
                output = self.linear(output)
                outputs.append(self.log_softmax(F.relu(output)))
            outputs = torch.stack(outputs)
        else:
            output_index = self.start_index
            while output_index != self.stop_index:
                output_embedding = self.embedding_layer(torch.tensor(output_index))
                output, (h_t, c_t) = self.decoder(torch.cat([output_embedding, additional_features]).view(1, 1, -1), (h_t, c_t))
                output = self.linear(output)
                output_index = output.argmax().item()
                outputs.append(output)

                if len(outputs) == self.max_output_length:
                    break
            outputs = torch.stack(outputs)
        return outputs

class MorphologicalInflector:
    def __init__(self, char2index, index2char, tag2index, embedding_size, hidden_size, max_output_length=25):
        """Initialises the class.

        Args:
            char2index: a dictionary which maps character to index
            index2char: a dictionary which maps indedx to character
            tag2index: a ditionary which maps morphological tag feature to index 
            embedding_size: embedding size
            hidden_size: size of hidden layer in encoder/decoder
            max_output_length: maximum length of decoder output
        """

        self.char2index = char2index
        self.index2char = index2char
        self.tag2index = tag2index

        self.encoder_decoder = EncoderDecoder(len(self.char2index), embedding_size, hidden_size,
                                             len(self.tag2index), self.char2index[START_CHAR],
                                             self.char2index[STOP_CHAR], 25)

    def words_to_tensor(self, words):
        """Converts list of words to a tensor with index

        Args:
            words: list of words

        Returns:
            tensor: a 2d tensor with each row containing indices for a sequence of characters
        """

        list_indices = []
        for word in words:
            word_indices = []
            for char in word:
                if char in self.char2index.keys():
                    word_indices.append(self.char2index[char])
                else:
                    word_indices.append(self.char2index[UNKNOWN_CHAR])
            list_indices.append(torch.tensor(word_indices))

        return list_indices

    def tag_to_vector(self, tags):
        """Returns one hot representation of tags given a tag.

        Args:
            tags: list of string representation of tag (eg, V;IND;PRS;2;PL)

        Returns: lsi
        """

        tag_vectors = []
        for tag in tags:
            tag_vector = torch.zeros(len(self.tag2index))
            for tag_feature in tag.split(';'):
                if tag_feature in self.tag2index:
                    tag_vector[self.tag2index[tag_feature]] = 1
                else:
                    tag_vector[self.tag2index[UNKNOWN_TAG]] = 1
            tag_vectors.append(tag_vector)
        return tag_vectors

    def indices_to_word(self, indices):
        """Returns a word given list contaning indices of words
        
        Args:
            indices: list containing indices

        Returns:
            word: a string
        """

        return ''.join([self.index2char[index] for index in indices])[:-1]
    
    def train(self, lemmas, tags, inflected_forms, epochs=1, learning_rate=0.01):
        """Trains the network.

        Args:
            lemmas: list of lemmas
            tags: list of tags
            inflected_forms: list of inflected_forms 
            epochs: numer of epochs to train for
            learning_rate: learning rate of the oprimiser

        Returns:
            nothing
        """

        criterion = nn.NLLLoss()
        optimizer = optim.Adadelta(self.encoder_decoder.parameters(), lr=learning_rate)

        t = trange(epochs)
        for epoch in t:
            epoch_loss = 0
            for lemma_tensor, tag_tensor, inflected_form_tensor in tqdm(zip(self.words_to_tensor(lemmas), self.tag_to_vector(tags), self.words_to_tensor(inflected_forms))):
                
                optimizer.zero_grad()
                loss = 0

                decoder_input = torch.cat([torch.tensor([self.char2index[START_CHAR]]), inflected_form_tensor])
                output = self.encoder_decoder(lemma_tensor, tag_tensor, decoder_input)
                output = output.squeeze(1).squeeze(1)

                target = torch.cat([inflected_form_tensor, torch.tensor([self.char2index[STOP_CHAR]])])
                loss = criterion(output, target)

                loss.backward()
                optimizer.step()

                epoch_loss += loss

            t.set_postfix(loss=epoch_loss.item() / len(lemmas))            

    def generate(self, lemmas, tags):
        """Generates output sequence.
        
        Args:
            lemmas: list of lemmas
            tags: list of morphological tags

        Returns:
            inflected_forms: list of inflected forms generated by the network.
        """

        inflected_forms = []

        for lemma_tensor, tag_tensor in zip(self.words_to_tensor(lemmas), self.tag_to_vector(tags)):

            output = self.encoder_decoder(lemma_tensor, tag_tensor)
            char_indices = output.argmax(dim=3).squeeze(1).squeeze(1).numpy().tolist()
            inflected_forms.append(self.indices_to_word(char_indices))

        return inflected_forms
            
if __name__ == "__main__":
    pass