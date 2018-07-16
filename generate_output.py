import os
import time

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm

from constants import TASK1_DATA_PATH
import dual_attention as package
from train import train_and_evaluate
from data import read_dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_languages(data_dir):
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


if __name__ == "__main__":
    languages = get_languages(TASK1_DATA_PATH)

    lr_dict = {'low': 0.001, 'medium': 0.001, 'high': 0.001}
    embedding_size_dict = {'low': 100, 'medium': 100, 'high': 300}
    hidden_size_dict = {'low': 100, 'medium': 100, 'high': 100}
    clip_dict = {'low': 3, 'medium': 3, 'high': 3}
    dropout_p_dict = {'low': 0.5, 'medium': 0.5, 'high': 0.3}
    alpha_dict = {'low': 0, 'medium': 0, 'high': 0}
    beta_dict = {'low': 0, 'medium': 0, 'high': 0}
    epochs_dict = {'low': 400, 'medium': 100, 'high': 20}

    output_folder = os.path.join('output', 'dh0p1')
    os.makedirs(output_folder)

    with open(os.path.join(output_folder, 'hyperparameters'), 'w', encoding='utf8') as file:
        file.write('lr: ' + str(lr_dict) + '\n' + 'embedding_size: ' + str(embedding_size_dict) + '\n' + 'hidden_size:' + str(hidden_size_dict)
                   + '\n' + 'clip: ' + str(clip_dict)+ '\n' + 'dropout_p_dict: ' + str(dropout_p_dict) + '\n'
                   + 'alpha: ' + str(alpha_dict) + '\n' + 'beta: ' + str(beta_dict) + '\n' + 'epochs: ' + str(epochs_dict))

    for language in tqdm(sorted(languages)):
        for dataset in ['low', 'medium', 'high']:

            lr = lr_dict[dataset]
            embedding_size = embedding_size_dict[dataset]
            hidden_size = hidden_size_dict[dataset]
            clip = clip_dict[dataset]
            dropout_p = dropout_p_dict[dataset]
            alpha = alpha_dict[dataset]
            beta = beta_dict[dataset]

            experiment_name = "dh0p1_{}_{}_lr{}_em{}_hd_{}_clip{}_p{}_a{}_b_{}_{}".format(language, dataset, lr, embedding_size, hidden_size, str(clip), dropout_p, alpha, beta, int(time.time()))

            try:
                model_inputs_train, model_inputs_val, labels_train, labels_val, vocab = package.data.load_data(language, dataset, test_data='dev', increase_val_data=True, test_size=0.2, random_state=42)
            except FileNotFoundError:
                continue

            model = package.net.Model(vocab, embedding_size=embedding_size, hidden_size=hidden_size,
                                      use_hierarchical_attention=False, use_ptr_gen=True,
                                      dropout_p=dropout_p).to(device)
            optimizer = optim.Adam(lr=lr, params=model.parameters())
            loss_fn = package.loss.Criterion(vocab, alpha, beta)

            writer = SummaryWriter('runs/' + experiment_name)
            model_save_dir = os.path.join('./saved_models', experiment_name)
            os.makedirs(model_save_dir)

            epochs = epochs_dict[dataset]
            train_and_evaluate(model_inputs_train, labels_train, model_inputs_val, labels_val, model, optimizer, loss_fn,
                                epochs=epochs, batch_size=32, model_save_dir=model_save_dir, print_output=False, writer=writer, clip=clip)

            # For prediction on dev dataset
            dev_file = os.path.join(TASK1_DATA_PATH, '{}-dev'.format(language))
            lemmas_dev, tags_dev, inflected_forms_dev = read_dataset(dev_file)

            # Load best performing model on validation set
            best_state = torch.load(os.path.join(model_save_dir, 'best.model'))
            model.load_state_dict(best_state['model_state'])

            file_path = os.path.join(output_folder, '{}-{}-out'.format(language, dataset))
            generate_output(model, lemmas_dev, tags_dev, file_path)
