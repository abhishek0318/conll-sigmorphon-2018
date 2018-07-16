import os

from constants import TASK1_DATA_PATH
from data import read_dataset
from utils import accuracy, average_distance

def mean(list):
    if len(list) != 0:
        return sum(list)/len(list)
    else:
        return 0.0

if __name__ == "__main__":
    predictions_dir = os.path.join('output', 'dh0p1')
    acc = {'low': [], 'medium': [], 'high': []}
    dist = {'low': [], 'medium': [], 'high': []}
    for filename in sorted(os.listdir(predictions_dir)):
        for dataset in ['low', 'medium', 'high']:
            if dataset in filename and dataset == filename.split('-')[-2]:
                language = '-'.join(filename.split('-')[:-2])
                _, _, predictions = read_dataset(os.path.join(predictions_dir, filename))
                _, _, truth = read_dataset(os.path.join(TASK1_DATA_PATH, '{}-dev'.format(language)))

                print('{}[task 1/{}]: {:.2f}, {:.2f}'.format(language, dataset, accuracy(predictions, truth), average_distance(predictions, truth)))
                acc[dataset].append(accuracy(predictions, truth))
                dist[dataset].append(average_distance(predictions, truth))

    print()
    print()
    for dataset in ['low', 'medium', 'high']:
        print('average[task 1/{}]: {:.2f}, {:.2f}'.format(dataset, mean(acc[dataset]), mean(dist[dataset])))
