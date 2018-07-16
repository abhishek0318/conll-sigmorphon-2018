import torch
import torch.nn as nn

from utils import accuracy, average_distance, pad_lists

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class OutputLoss(nn.Module):
    def __init__(self, vocab):
        super(OutputLoss, self).__init__()
        self.vocab = vocab
        self.criterion = nn.NLLLoss(ignore_index=-1)

    def forward(self, p_ws, inflected_forms_indices):
        max_decoder_len = p_ws.shape[1]
        tgt_classes = p_ws.shape[2]
        inflected_forms_indices = [seq_indices + [self.vocab.char_to_index(self.vocab.STOP_CHAR)] for seq_indices in inflected_forms_indices]
        p_ws_target = pad_lists(inflected_forms_indices, -1, pad_len=max_decoder_len, dtype=torch.long, device=device)
        loss = self.criterion(p_ws.view(-1, tgt_classes), p_ws_target.view(-1))
        return loss


class AlignmentLoss(nn.Module):
    def __init__(self):
        super(AlignmentLoss, self).__init__()
        self.criterion = nn.NLLLoss(ignore_index=-1)

    def forward(self, a_ls, a_ls_true):
        max_decoder_len = a_ls.shape[1]
        max_lemma_len = a_ls.shape[2]
        target = pad_lists(a_ls_true, -1, pad_len=max_decoder_len, dtype=torch.long, device=device)
        loss = self.criterion(torch.log(a_ls + 1e-6).view(-1, max_lemma_len), target.view(-1))
        return loss


class PGenLoss(nn.Module):
    def __init__(self):
        super(PGenLoss, self).__init__()
        self.criterion = nn.NLLLoss(ignore_index=-1)

    def forward(self, p_gens, p_gens_true):
        bsz = p_gens.shape[0]
        max_decoder_len = p_gens.shape[1]
        score = torch.zeros(bsz, max_decoder_len, 2, device=device)
        score[:, :, 0] = 1 - p_gens
        score[:, :, 1] = p_gens
        target = pad_lists(p_gens_true, -1, pad_len=max_decoder_len, dtype=torch.long, device=device)
        loss = self.criterion(torch.log(score + 1e-6).view(-1, 2), target.view(-1))
        return loss


class OutputLoss(nn.Module):
    def __init__(self, vocab):
        super(OutputLoss, self).__init__()
        self.vocab = vocab
        self.criterion = nn.NLLLoss(ignore_index=-1)

    def forward(self, p_ws, inflected_forms_indices):
        max_decoder_len = p_ws.shape[1]
        tgt_classes = p_ws.shape[2]
        inflected_forms_indices = [seq_indices + [self.vocab.char_to_index(self.vocab.STOP_CHAR)] for seq_indices in inflected_forms_indices]
        p_ws_target = pad_lists(inflected_forms_indices, -1, pad_len=max_decoder_len, dtype=torch.long, device=device)
        loss = self.criterion(p_ws.view(-1, tgt_classes), p_ws_target.view(-1))
        return loss


class Criterion(nn.Module):

    def __init__(self, vocab, alpha, beta):
        super(Criterion, self).__init__()
        self.alignment_loss = AlignmentLoss()
        self.output_loss = OutputLoss(vocab)
        self.p_gen_loss = PGenLoss()
        self.vocab = vocab
        self.alpha = alpha
        self.beta = beta

    def forward(self, p_ws, a_ls, p_gens, inflected_forms_indices, a_ls_true, p_gens_true):

        o_loss = self.output_loss(p_ws, inflected_forms_indices)
        a_loss = self.alignment_loss(a_ls, a_ls_true)
        p_loss = self.p_gen_loss(p_gens, p_gens_true)
        loss = o_loss + self.alpha * a_loss + self.beta * p_loss

        inflected_forms_predicted, inflected_forms_truth = self.to_word(p_ws, inflected_forms_indices)
        loss_dict = {}
        loss_dict['loss'] = loss.item()
        loss_dict['acc'] = accuracy(inflected_forms_predicted, inflected_forms_truth) * len(inflected_forms_indices)
        loss_dict['dist'] = average_distance(inflected_forms_predicted, inflected_forms_truth) * len(inflected_forms_indices)
        loss_dict['o_loss'] = o_loss.item()
        loss_dict['a_loss'] = a_loss.item()
        loss_dict['p_loss'] = p_loss.item()

        return loss, loss_dict

    def to_word(self, p_ws, indices):
        predictions = [word.split(self.vocab.STOP_CHAR)[0] for word in self.vocab.indices_to_word(p_ws.argmax(2))]
        true_values = self.vocab.indices_to_word(indices)
        return predictions, true_values