"""Notation as used in 'Get To The Point: Summarization with Pointer-Generator Networks'
 https://arxiv.org/pdf/1704.04368.pdf"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import pad_lists

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        """Bahdanau Attention as described in Pointer-Generator Networks paper."""

        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.W_h = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.W_s = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_attn = nn.Parameter(torch.Tensor(hidden_size)) # https://pytorch.org/docs/master/_modules/torch/nn/modules/linear.html#Linear
        self.init_parameters()

    def forward(self, h, s, mask):
        """

        Args:
            h: encoder outputs, of shape (bsz, max_src_len, 2*hidden_size)
            s: decoder hidden state, of shape (1, bsz, hidden_size)
            mask: ByteTensor, of shape (bsz, max_src_len), 1 where source is 0

        Returns:
            a: attention weights, of shape (bsz, max_src_len)
        """

        bsz = h.shape[0]
        max_src_len = h.shape[1]

        e = self.W_h(h) + self.W_s(s.transpose(0, 1).expand(-1, max_src_len, -1)) + self.b_attn.expand(bsz, max_src_len, -1) # (bsz, max_src_len, hidden_size)
        e = F.tanh(e)  # (bsz, max_src_len, hidden_size)
        e = self.v(e).squeeze(2)  # (bsz, max_src_len)
        e.data.masked_fill_(mask, float('-inf'))  # (bsz, max_src_len)
        a = F.softmax(e, dim=1)  # (bsz, max_src_len)

        return a

    def init_parameters(self):
        # https://pytorch.org/docs/master/_modules/torch/nn/modules/linear.html#Linear
        stdv = 1 / math.sqrt(self.hidden_size)
        self.b_attn.data.uniform_(-stdv, stdv)


class HierarchicalAttention(nn.Module):
    """Based on Hierarchical Attention described here - https://arxiv.org/pdf/1704.06567.pdf"""

    def __init__(self, hidden_size):
        super(HierarchicalAttention, self).__init__()

        self.hidden_size = hidden_size
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.W_h_l = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.W_h_tg = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.W_s = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_attn = nn.Parameter(torch.Tensor(hidden_size))  # https://pytorch.org/docs/master/_modules/torch/nn/modules/linear.html#Linear
        self.init_parameters()

    def forward(self, h_l, h_tg, s):
        """

        Args:
            h_l: context vector over lemma, of shape (bsz, 2*hidden_size)
            h_tg: context vector over tags, of shape (bsz, 2*hidden_size)
            s: decoder hidden state, of shape (1, bsz, hidden_size)

        Returns:
            a: attention weight over lemma context vector, of shape (bsz, )

        """

        bsz = h_l.shape[0]

        e = self.W_h_l(h_l) + self.W_h_tg(h_tg) + self.W_s(s).squeeze(0) + self.b_attn.expand(bsz, -1)  # (bsz, hidden_size)
        e = F.tanh(e)
        e = self.v(e).squeeze(1)  # (bsz, )
        a = F.sigmoid(e)

        return a

    def init_parameters(self):
        # https://pytorch.org/docs/master/_modules/torch/nn/modules/linear.html#Linear
        stdv = 1 / math.sqrt(self.hidden_size)
        self.b_attn.data.uniform_(-stdv, stdv)


class GenerationProbabilty(nn.Module):
    def __init__(self, embedding_size, hidden_size, h_star_size):
        """Calculates `p_gen` as described in Pointer-Generator Networks paper."""

        super(GenerationProbabilty, self).__init__()
        self.W_h_star = nn.Linear(h_star_size, 1, bias=False)
        self.W_s = nn.Linear(hidden_size, 1, bias=False)
        self.W_x = nn.Linear(embedding_size, 1, bias=False)
        self.b_attn = nn.Parameter(torch.Tensor(1))  # https://pytorch.org/docs/master/_modules/torch/nn/modules/linear.html#Linear
        self.init_parameters()

    def forward(self, h_star, s, x):
        """

        Args:
            h_star: combined context vector over lemma and tag
            s: decoder hidden state, of shape (1, bsz, hidden_size)
            x: decoder input, of shape (bsz, embedding_size)

        Returns:
            p_gen: generation probabilty, of shape (bsz, )
        """

        bsz = h_star.shape[0]
        p_gen = self.W_h_star(h_star) + self.W_s(s.squeeze(0)) + self.W_x(x) + self.b_attn.expand(bsz, -1)  # (bsz, 1)
        p_gen = F.sigmoid(p_gen.squeeze(1))  # (bsz, )

        return p_gen

    def init_parameters(self):
        # https://pytorch.org/docs/master/_modules/torch/nn/modules/linear.html#Linear
        stdv = 1 / math.sqrt(100)  # !!!
        self.b_attn.data.uniform_(-stdv, stdv)


class Encoder(nn.Module):
    def __init__(self, vocab, vocab_size, embedding_size, hidden_size, dropout_p=0):
        super(Encoder, self).__init__()

        self.vocab = vocab
        self.embedder = nn.Embedding(vocab_size, embedding_size, padding_idx=vocab.padding_idx)
        self.dropout_input = nn.Dropout(p=dropout_p)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout_output = nn.Dropout(p=dropout_p)

    def forward(self, indices):
        """

        Args:
            indices: list containing sequences of indices, of length bsz

        Returns:
            h: hidden state at each time step, of shape (bsz, max_src_len, 2*hidden_size)
            mask: 1 where input index is 0 (bsz, max_src_len)
            (h_n, c_n): final hidden state, a tuple ((1, bsz, hidden_size), (1, bsz, hidden_size))
        """

        # Inspired from here, https://discuss.pytorch.org/t/rnns-sorting-operations-autograd-safe/1461
        # See also, https://discuss.pytorch.org/t/solved-multiple-packedsequence-input-ordering/2106

        lengths = torch.tensor([len(x) for x in indices], dtype=torch.long, device=device)
        indices_padded = pad_lists(indices, self.vocab.padding_idx, dtype=torch.long, device=device)
        lengths_sorted, sorted_idx = lengths.sort(descending=True)
        indices_sorted = indices_padded[sorted_idx]
        embeddings_padded = self.embedder(indices_sorted)
        embeddings_padded = self.dropout_input(embeddings_padded)
        embeddings_packed = pack_padded_sequence(embeddings_padded, lengths_sorted.tolist(), batch_first=True)
        h, (h_n, c_n) = self.lstm(embeddings_packed)
        h, _ = pad_packed_sequence(h, batch_first=True, padding_value=self.vocab.padding_idx)
        h = torch.zeros_like(h).scatter_(0, sorted_idx.unsqueeze(1).unsqueeze(1).expand(-1, h.shape[1], h.shape[2]), h)  # Revert sorting
        h_n = torch.zeros_like(h_n).scatter_(1, sorted_idx.unsqueeze(0).unsqueeze(2).expand(h_n.shape[0], -1, h_n.shape[2]), h_n)  # Revert sorting
        c_n = torch.zeros_like(c_n).scatter_(1, sorted_idx.unsqueeze(0).unsqueeze(2).expand(c_n.shape[0], -1, c_n.shape[2]), c_n)  # Revert sorting
        h = self.dropout_output(h)
        h_n = (h_n[0, :, :] + h_n[1, :, :]).unsqueeze(0)  # (1, bsz, hidden_size)
        c_n = (c_n[0, :, :] + c_n[1, :, :]).unsqueeze(0)  # (1, bsz, hidden_size)
        mask = indices_padded == 0  # (bsz, max_lemma_len)

        return h, mask, (h_n, c_n)


class Decoder(nn.Module):
    def __init__(self, vocab, embedding_size, hidden_size, use_hierarchical_attention, use_ptr_gen, max_decode_len=25, epsilon=1e-6, dropout_p=0):
        super(Decoder, self).__init__()
        self.lemma_embedder = nn.Embedding(vocab.char_vocab_size, embedding_size, padding_idx=vocab.padding_idx)
        self.vocab = vocab
        self.max_decode_len = max_decode_len
        self.hidden_size = hidden_size
        self.use_hierarchical_attention = use_hierarchical_attention
        self.use_ptr_gen = use_ptr_gen
        self.epsilon = epsilon

        self.dropout = nn.Dropout(dropout_p)
        if use_hierarchical_attention:
            self.lstm = nn.LSTM(embedding_size + 2*hidden_size, hidden_size, batch_first=True)
            self.hierarchical_attention = HierarchicalAttention(hidden_size)
            if use_ptr_gen:
                self.generation_probability = GenerationProbabilty(embedding_size, hidden_size, 2*hidden_size)
            self.generator = nn.Linear(3*hidden_size, vocab.char_vocab_size)
        else:
            self.lstm = nn.LSTM(embedding_size + 4*hidden_size, hidden_size, batch_first=True)
            if use_ptr_gen:
                self.generation_probability = GenerationProbabilty(embedding_size, hidden_size, 4*hidden_size)
            self.generator = nn.Linear(5*hidden_size, vocab.char_vocab_size)

        self.lemma_attention = BahdanauAttention(hidden_size)
        self.tag_attention = BahdanauAttention(hidden_size)

    def forward(self, lemma_input_indices, h_l, h_tg, mask_l, mask_tg, decoder_initial, inputs=None, a_ls_true=None, p_gens_true=None):
        """

        Args:
            lemma_input_indices: (bsz, max_lemma_len) for p_gen
            h_l: (bsz, max_lemma_len, 2*hidden_size)
            h_tg: (bsz, max_tag_len, 2*hidden_size)
            mask_l: (bsz, max_lemma_len)
            mask_tg: (bsz, max_tag_len)
            decoder_initial: (1, bsz, hidden_size), (1, bsz, hidden_size)
            inputs: input to decoder for teacher forcing, of shape (bsz, max_len, embedding_size)
            a_ls_true: true alignments for teacher forcing
            p_gens_true: true p_gens for teacher forcing

        Returns:
            p_ws: log probabilities, of shape (bsz, max_decode_len, char_vocab_size)
            a_ls: attention over lemmas, of shape (bsz, max_decode_len, max_lemma_len)
            p_gens: p_gens, of shape (bsz, max_decode_len)
        """

        bsz = h_l.shape[0]
        max_lemma_len = h_l.shape[1]

        max_t = self.max_decode_len
        if inputs is not None:
            max_t = min(max_t, inputs.shape[1])
        else:
            x = torch.ones(bsz, device=device, dtype=torch.long) * self.vocab.char_to_index(self.vocab.START_CHAR) # (bsz, )
            x = self.lemma_embedder(x)  # (bsz, embedding_size)

        s, c = decoder_initial  # (1, bsz, hidden_size), (1, bsz, hidden_size)

        p_ws = torch.ones(bsz, self.max_decode_len, self.vocab.char_vocab_size, device=device) * self.epsilon  # (bsz, max_decode_len, char_vocab_size)
        a_ls = torch.zeros(bsz, self.max_decode_len, max_lemma_len, device=device)  # (bsz, max_decode_len, max_lemma_len)
        p_gens = torch.zeros(bsz, self.max_decode_len, device=device)  # (bsz, max_decode_len)

        for t in range(max_t):
            if inputs is not None:
                x = inputs[:, t, :]  # (bsz, embedding_size)
            x = self.dropout(x)
            a_l = self.lemma_attention(h_l, s, mask_l)  # (bsz, max_lemma_len)
            a_tg = self.tag_attention(h_tg, s, mask_tg)  # (bsz, max_tag_len)

            # (bsz, 1, max_lemma_len) X (bsz, max_lemma_len, 2*hidden_size) -> (bsz, 1, 2*hidden_size)
            h_l_star = torch.bmm(a_l.unsqueeze(1), h_l).squeeze(1)   # (bsz, 2*hidden_size)
            # (bsz, 1, max_tag_len) X (bsz, max_tag_len, 2*hidden_size) -> (bsz, 1, 2*hidden_size)
            h_tg_star = torch.bmm(a_tg.unsqueeze(1), h_tg).squeeze(1)  # (bsz, 2*hidden_size)

            # combine two context vectors
            if self.use_hierarchical_attention:
                a_h = self.hierarchical_attention(h_l_star, h_tg_star, s)  # (bsz, )

                # https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
                # (bsz, 1) * (bsz, 2*hidden_size) -> (bsz, 2*hidden_size)
                h_c_star = a_h.unsqueeze(1) * h_l_star + (1 - a_h).unsqueeze(1) * h_tg_star  # (bsz, 2*hidden_size)
            else:
                h_c_star = torch.cat([h_l_star, h_tg_star], dim=1)  # (bsz, 4*hidden_size)

            _, (s, c) = self.lstm(torch.cat([x, h_c_star], dim=1).unsqueeze(1), (s, c))

            p_vocab = F.softmax(self.generator(torch.cat([s.squeeze(0), h_c_star], dim=1)), dim=1)  # (bsz, char_vocab_size)

            if self.use_ptr_gen:
                p_attn = torch.zeros(bsz, self.vocab.char_vocab_size, device=device)  # (bsz, char_vocab_size)
                p_attn.scatter_add_(1, lemma_input_indices, a_l)

                p_gen = self.generation_probability(h_c_star, s, x)  # (bsz, )

                # https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
                # (bsz, 1) * (bsz, 2*char_vocab_size) -> (bsz, 2*char_vocab_size)
                p_w = p_gen.unsqueeze(1) * p_vocab + (1 - p_gen).unsqueeze(1) * p_attn  # (bsz, char_vocab_size)
            else:
                p_w = p_vocab

            x = self.lemma_embedder(p_w.argmax(dim=1))  # (bsz, embedding_size)
            p_ws[:, t, :] = p_w
            a_ls[:, t, :] = a_l
            if self.use_ptr_gen:
                p_gens[:, t] = p_gen

        return torch.log(p_ws), a_ls, p_gens


class Model(nn.Module):
    def __init__(self, vocab, embedding_size, hidden_size, use_hierarchical_attention, use_ptr_gen, dropout_p=0):
        super(Model, self).__init__()
        self.vocab = vocab
        self.lemma_encoder = Encoder(vocab, vocab.char_vocab_size, embedding_size, hidden_size, dropout_p=dropout_p)
        self.tag_encoder = Encoder(vocab, vocab.tag_vocab_size, embedding_size, hidden_size, dropout_p=dropout_p)
        self.bridge_h = nn.Linear(2*hidden_size, hidden_size)
        self.bridge_c = nn.Linear(2*hidden_size, hidden_size)
        self.decoder = Decoder(vocab, embedding_size, hidden_size, use_hierarchical_attention, use_ptr_gen, dropout_p=dropout_p)
        self.decoder.lemma_embedder.weight = self.lemma_encoder.embedder.weight  # share weights

    def forward(self, lemma_indices, tag_indices, inflected_form_indices=None, a_ls_true=None, p_gens_true=None):
        """

        Args:
            lemma_indices: list of list containing lemma indices
            tag_indices: list of list containing tag indices
            inflected_form_indices: list of list containing inflected form indices (for teacher forcing)
            a_ls_true: true alignments (for teacher forcing)
            p_gens_true: true p_gens (for teacher forcing)

        Returns:
            p_ws: log probabilities, of shape (bsz, max_decode_len, char_vocab_size)
            a_ls: attention over lemmas, of shape (bsz, max_decode_len, max_lemma_len)
            p_gens: p_gens, of shape (bsz, max_decode_len)
        """

        # (bsz, max_lemma_len, 2*hidden_size), (bsz, max_lemma_len), (1, bsz, hidden_size)
        h_l, mask_l, (h_l_n, c_l_n) = self.lemma_encoder(lemma_indices)

        # (bsz, max_tag_len, 2*hidden_size), (bsz, max_tag_len), (1, bsz, hidden_size)
        h_tg, mask_tg, (h_tg_n, c_tg_n) = self.tag_encoder(tag_indices)

        # (1, bsz, hidden_size) & (1, bsz, hidden_size) -> (1, bsz, hidden_size)
        s_0 = self.bridge_h(torch.cat([h_l_n, h_tg_n], dim=2))

        # (1, bsz, hidden_size) & (1, bsz, hidden_size) -> (1, bsz, hidden_size)
        c_0 = self.bridge_c(torch.cat([c_l_n, c_tg_n], dim=2))

        lemma_indices_padded = pad_lists(lemma_indices, self.vocab.padding_idx, dtype=torch.long, device=device)

        if inflected_form_indices is not None:
            inflected_form_indices = [[self.vocab.char_to_index(self.vocab.START_CHAR)] + seq_indices for seq_indices in inflected_form_indices]
            inflected_form_indices = pad_lists(inflected_form_indices, self.vocab.padding_idx, dtype=torch.long, device=device) # (bsz, max_tgt_len)
            decoder_input = self.lemma_encoder.embedder(inflected_form_indices)  # (bsz, max_tgt_len, embedding_size)
        else:
            decoder_input = None

        p_ws, a_ls, p_gens = self.decoder(lemma_indices_padded, h_l, h_tg, mask_l, mask_tg, (s_0, c_0), decoder_input, a_ls_true, p_gens_true)

        return p_ws, a_ls, p_gens
