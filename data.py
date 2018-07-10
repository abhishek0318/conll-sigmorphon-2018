def read_dataset(file_name):
    """Reads dataset.

    Args:
        file_name: path to file containing the data

    Returns:
        lemmas: list of lemma
        tags: list of tags
        inflected_forms: list of inflected form
    """

    with open(file_name, 'r', encoding='utf') as file:
        text = file.read()

    lemmas = []
    tags = []
    inflected_forms = []

    for line in text.split('\n')[:-1]:
        lemma, inflected_form, tag = line.split('\t')
        lemmas.append(lemma)
        inflected_forms.append(inflected_form)
        tags.append(tag)

    return lemmas, tags, inflected_forms

