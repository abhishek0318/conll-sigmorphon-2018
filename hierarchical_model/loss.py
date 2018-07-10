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


class Criterion(nn.Module):

    def __init__(self, vocab):
        super(Criterion, self).__init__()
        self.output_loss = OutputLoss(vocab)
        self.vocab = vocab

    def forward(self, p_ws, a_ls, p_gens, inflected_forms_indices, a_ls_true, p_gens_true):

        loss = self.output_loss(p_ws, inflected_forms_indices)

        inflected_forms_predicted, inflected_forms_truth = self.to_word(p_ws, inflected_forms_indices)
        loss_dict = {}
        loss_dict['loss'] = loss.item()
        loss_dict['acc'] = accuracy(inflected_forms_predicted, inflected_forms_truth) * len(inflected_forms_indices)
        loss_dict['dist'] = average_distance(inflected_forms_predicted, inflected_forms_truth) * len(inflected_forms_indices)

        return loss, loss_dict

    def to_word(self, p_ws, indices):
        predictions = [word.split(self.vocab.STOP_CHAR)[0] for word in self.vocab.indices_to_word(p_ws.argmax(2))]
        true_values = self.vocab.indices_to_word(indices)
        return predictions, true_values