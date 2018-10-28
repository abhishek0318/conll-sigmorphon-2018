import os
import time

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm

from constants import TASK1_DATA_PATH
import dual_attention as package
from train import train_and_evaluate
from data import read_dataset, read_covered_dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_languages():
    languages = []
    for filename in os.listdir(TASK1_DATA_PATH):
        if 'train' in filename:
            languages.append(filename.split('-train-')[0])
    return set(languages)


def generate_output(model, lemmas, tags, file_path):
    predictions = package.data.generate_predictions(model, lemmas, tags)
    with open(file_path, 'w', encoding='utf8') as file:
        for lemma, inflected_form, tag in zip(lemmas, predictions, tags):
            file.write(lemma + '\t' + inflected_form + '\t' + tag + '\n')


def generate_entry(model_name, hyperparameters, datasets=('low', 'medium', 'high'), use_hierarchical_attention=False, use_ptr_gen=True, test_data='test', write_hyperparameter=False, output_folder=None, resume=False):

    languages = get_languages()

    if output_folder is None:
        output_folder = os.path.join('output', model_name)
    if not resume:
        os.makedirs(output_folder)

    if write_hyperparameter:
        with open(os.path.join(output_folder, 'hyperparameters'), 'w', encoding='utf8') as file:
            file.write(hyperparameters)

    for language in tqdm(sorted(languages)[60:]):
        for dataset in datasets:
            if resume and os.path.exists(os.path.join(output_folder, '{}-{}-out'.format(language, dataset))):
                continue
            lr = hyperparameters['lr'][dataset]
            embedding_size = hyperparameters['embedding_size'][dataset]
            hidden_size = hyperparameters['hidden_size'][dataset]
            clip = hyperparameters['clip'][dataset]
            dropout_p = hyperparameters['dropout_p'][dataset]
            alpha = hyperparameters['alpha'][dataset]
            beta = hyperparameters['beta'][dataset]
            patience = hyperparameters['patience'][dataset]
            epochs_extension = hyperparameters['epochs_extension'][dataset]

            experiment_name = "{}_{}_{}_lr{}_em{}_hd_{}_clip{}_p{}_a{}_b_{}_{}".format(model_name, language, dataset, lr, embedding_size, hidden_size, str(clip), dropout_p, alpha, beta, int(time.time()))

            try:
                model_inputs_train, model_inputs_val, labels_train, labels_val, \
                vocab = package.data.load_data(language, dataset, test_data=test_data, use_external_val_data=True,
                                               val_ratio=0.2, random_state=42)
            except FileNotFoundError:
                continue

            model = package.net.Model(vocab, embedding_size=embedding_size, hidden_size=hidden_size,
                                      use_hierarchical_attention=use_hierarchical_attention, use_ptr_gen=use_ptr_gen,
                                      dropout_p=dropout_p).to(device)
            optimizer = optim.Adam(lr=lr, params=model.parameters())
            loss_fn = package.loss.Criterion(vocab, alpha, beta)

            writer = SummaryWriter('runs/' + experiment_name)
            model_save_dir = os.path.join('./saved_models', experiment_name)
            os.makedirs(model_save_dir)

            epochs = hyperparameters['epochs'][dataset]
            train_and_evaluate(model_inputs_train, labels_train, model_inputs_val, labels_val, model, optimizer, loss_fn,
                               epochs=epochs, batch_size=32, model_save_dir=model_save_dir, show_progress=False, writer=writer, clip=clip)
            epochs_trained = epochs

            # Load best performing model on validation set
            best_state = torch.load(os.path.join(model_save_dir, 'best.model'))
            while epochs_trained - best_state['epoch_num'] < patience:
                train_and_evaluate(model_inputs_train, labels_train, model_inputs_val, labels_val, model, optimizer,
                                   loss_fn, epochs=epochs_extension, batch_size=32, model_save_dir=model_save_dir, show_progress=False,
                                   writer=writer, clip=clip, starting_epoch=epochs_trained+1, initial_best_val_acc=best_state['val_acc'])
                epochs_trained += epochs_extension
                best_state = torch.load(os.path.join(model_save_dir, 'best.model'))
            model.load_state_dict(best_state['model_state'])

            if test_data == 'dev':
                dev_file = os.path.join(TASK1_DATA_PATH, '{}-dev'.format(language))
                lemmas_test, tags_test, _ = read_dataset(dev_file)
            elif test_data == 'test':
                test_file = os.path.join(TASK1_DATA_PATH, '{}-covered-test'.format(language))
                lemmas_test, tags_test = read_covered_dataset(test_file)
            else:
                raise ValueError

            file_path = os.path.join(output_folder, '{}-{}-out'.format(language, dataset))
            generate_output(model, lemmas_test, tags_test, file_path)


if __name__ == "__main__":
    hyperparameters = {
        'lr': {'low': 0.001, 'medium': 0.001, 'high': 0.001},
        'embedding_size': {'low': 100, 'medium': 100, 'high': 300},
        'hidden_size': {'low': 100, 'medium': 100, 'high': 100},
        'clip': {'low': 3, 'medium': 3, 'high': 3},
        'dropout_p': {'low': 0.5, 'medium': 0.5, 'high': 0.3},
        'alpha': {'low': 0, 'medium': 0, 'high': 0},
        'beta': {'low': 0, 'medium': 0, 'high': 0},
        'epochs': {'low': 300, 'medium': 80, 'high': 60},
        'patience': {'low': 100, 'medium': 20, 'high': 10},
        'epochs_extension': {'low': 100, 'medium': 20, 'high': 10}
    }
    generate_entry('dh0p1', hyperparameters, datasets=('high', ), use_hierarchical_attention=False, use_ptr_gen=True, test_data='test', resume=True)

