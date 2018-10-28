"""Script to train a network for particular language and dataset pair."""

import os
import time

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter
from tqdm import trange

from constants import *
import dual_attention as package
from utils import add_dict, add_string_to_key, divide_dict, grouper, save_model, shuffle_together

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_and_evaluate(input_train, labels_train, input_val, labels_val, model, optimizer, loss_fn, epochs=1, batch_size=32, clip=None, show_progress=True, writer=None, model_save_dir=None, starting_epoch=1, initial_best_val_acc=-0.01):
    """Trains and tests the model (on the validation set). Additionally saves the model with best val accuracy.

    Args:
        input_train: list of tuples containing model input for training
        labels_train: list of tuples containing labels corresponding to model input for training
        input_val: list of tuples containing model input for validation
        labels_val list of tuples containing labels corresponding to model input for validation
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        epochs: number of epochs to run
        batch_size: maximum batch_size
        clip: the value to which clip the norm of gradients to
        writer: tensorboardX.SummaryWriter
        model_save_dir: directory where to save the model
        starting_epoch: epoch number to start with.
        initial_best_val_acc: initial best accuracy on validation set, useful if resuming training.
    """

    best_val_acc = initial_best_val_acc
    if show_progress:
        t = trange(starting_epoch, starting_epoch + epochs)
    else:
        t = range(starting_epoch, starting_epoch + epochs)

    for epoch_num in t:
        # shuffle training data
        input_train, labels_train = shuffle_together(input_train, labels_train)

        epoch_metrics = {} # dictionary which stores metrics for a particular epoch.
        for batch_input, batch_labels in zip(grouper(input_train, batch_size), grouper(labels_train, batch_size)):
            batch_input = list(filter(lambda x: x is not None, batch_input))  # remove None objects introduced by grouper
            batch_labels = list(filter(lambda x: x is not None, batch_labels))  # remove None objects introduced by grouper

            batch_metrics = train(batch_input, batch_labels, model, optimizer, loss_fn, clip)
            add_dict(epoch_metrics, batch_metrics)

        val_metrics = test(input_val, labels_val, model, loss_fn, batch_size)

        epoch_metrics = divide_dict(epoch_metrics, len(input_train))
        val_metrics = divide_dict(val_metrics, len(input_val))

        # Saves best model till now.
        if model_save_dir:
            if val_metrics['acc'] > best_val_acc:
                best_val_acc = val_metrics['acc']
                save_model(epoch_num, {'epoch_num': epoch_num, 'val_acc': val_metrics['acc'],
                                       'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()},
                           model_save_dir, filename='best.model')

        metrics = {**add_string_to_key(epoch_metrics, 'train'), **add_string_to_key(val_metrics, 'val')}

        # Write to tensorboard
        if writer is not None:
            for key, value in metrics.items():
                writer.add_scalar(key, value, epoch_num)

        if show_progress:
            t.set_postfix(metrics)


def train(model_input, labels, model, optimizer, loss_fn, clip=None):
    """Train the model on `num_steps` batches

    Args:
        model_input: list of tuples containing model input
        labels: list of tuples containing labels corresponding to model input for training
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        clip: the value to which clip the norm of gradients to

    Returns:
        loss_dict: dictionary containing metrics
    """

    # set model to training mode
    model.train()

    # compute model output and loss
    model_out = model(*zip(*model_input), *zip(*labels))
    loss, loss_dict = loss_fn(*model_out, *zip(*labels))

    # clear previous gradients, compute gradients of all variables wrt loss
    optimizer.zero_grad()
    loss.backward()

    # performs updates using calculated gradients
    if clip:
        clip_grad_norm_(model.parameters(), clip)
    optimizer.step()

    return loss_dict


def test(model_input, labels, model, loss_fn=None, batch_size=32):
    """

    Args:
        model_input: list of tuples containing input to model
        labels: list of tuples containing labels corresponding to model input for training
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        batch_size: maximum batch_size

    Returns:
        metrics: dict
    """

    metrics = {}
    for batch_input, batch_labels in zip(grouper(model_input, batch_size), grouper(labels, batch_size)):
        batch_input = list(filter(lambda x: x is not None, batch_input))  # remove None objects introduced by grouper
        batch_labels = list(filter(lambda x: x is not None, batch_labels))  # remove None objects introduced by grouper

        batch_metrics = test_batch(batch_input, batch_labels, model, loss_fn=loss_fn)
        add_dict(metrics, batch_metrics)

    return metrics


def test_batch(model_input, labels, model, loss_fn=None):
    """Test the model on `num_steps` batches.

    Args:
        model_input: list of tuples containing input to model
        labels: list of tuples containing labels corresponding to model input
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch

    Returns:
        metrics: dictionary containing metrics if loss_fn given, else returns empty dictionary
    """

    # set model to evaluating mode
    model.eval()

    # compute model output and loss
    model_out = model(*zip(*model_input))

    if loss_fn:
        loss, metrics = loss_fn(*model_out, *zip(*labels))
        return metrics
    else:
        return {}


if __name__ == "__main__":
    language = 'hindi'
    dataset = 'low'
    lr = 0.001
    embedding_size = 100
    hidden_size = 100
    clip = 3
    dropout_p = 0.5
    alpha = 0 # coefficient of attention loss
    beta = 0 # coefficient of p_gen loss

    experiment_name = "sp0_{}_{}_lr{}_em{}_hd_{}_clip{}_p{}_a{}_b{}_{}".format(language, dataset, lr, embedding_size,
                                                                              hidden_size, str(clip), dropout_p, alpha,
                                                                              beta, int(time.time()))

    model_inputs_train, model_inputs_val, labels_train, labels_val, vocab = package.data.load_data(language, dataset,
                                                                                                   test_data='dev',
                                                                                                   use_external_val_data=True,
                                                                                                   val_ratio=0.2,
                                                                                                   random_state=42)

    model = package.net.Model(vocab, embedding_size=embedding_size, hidden_size=hidden_size, use_ptr_gen=False, dropout_p=dropout_p).to(device)
    optimizer = optim.Adam(lr=lr, params=model.parameters())
    loss_fn = package.loss.Criterion(vocab, alpha, beta)

    writer = SummaryWriter('runs/' + experiment_name)
    model_save_dir = os.path.join('./saved_models', experiment_name)
    os.makedirs(model_save_dir)
    model_save_dir = None
    try:
        train_and_evaluate(model_inputs_train, labels_train, model_inputs_val, labels_val, model, optimizer, loss_fn,
                        epochs=400, batch_size=32, model_save_dir=model_save_dir, writer=writer, clip=clip)
    except KeyboardInterrupt:
        result_text, accuracy, distance = package.data.evaluate_on_dev(model, os.path.join(TASK1_DATA_PATH, '{}-dev'.format(language)))
        print(result_text, '\n', '{}, {}'.format(accuracy, distance))
    else:
        result_text, accuracy, distance = package.data.evaluate_on_dev(model, os.path.join(TASK1_DATA_PATH, '{}-dev'.format(language)))
        print(result_text, '\n', '{}, {}'.format(accuracy, distance))