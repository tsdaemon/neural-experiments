import subprocess
import argparse
import os

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-s', help='ISO 639-3 code for a source language')
parser.add_argument('-t', help='ISO 639-3 code for a target language')
parser.add_argument('-data_dir', default='./data', help='Data directory')


if __name__ == '__main__':
    args = parser.parse_args()
    source_lang = args.s
    target_lang = args.t
    data_dir = args.data_dir

    # download if sentences are not here
    sentences_path = os.path.join(data_dir, 'sentences.csv')
    if not os.path.isfile(sentences_path):
        if not os.path.isfile(os.path.join(data_dir, 'sentences.tar.bz2')):
            subprocess.run(
                "wget https://downloads.tatoeba.org/exports/sentences.tar.bz2 -P " + data_dir,
                shell=True)

        subprocess.run(
            "tar xvjC {0} -f {0}/sentences.tar.bz2".format(data_dir), shell=True)

    # download if links are not here
    links_path = os.path.join(data_dir, 'links.csv')
    if not os.path.isfile(links_path):
        if not os.path.isfile(os.path.join(data_dir, 'links.tar.bz2')):
            subprocess.run(
                "wget https://downloads.tatoeba.org/exports/links.tar.bz2 -P " + data_dir,
                shell=True)

        subprocess.run("tar xvjC {0} -f {0}/links.tar.bz2".format(data_dir), shell=True)

    # read all data
    sentences = pd.read_csv(sentences_path, names=['id', 'lang', 'text'], header=None, delimiter='\t')
    links = pd.read_csv(links_path, names=['sent_id', 'tran_id'], header=None, delimiter='\t')

    # extract source - target connected
    source_sentences = sentences[sentences.lang == source_lang]
    source_sentences = source_sentences.merge(links, left_on='id', right_on='sent_id')
    target_sentences = sentences[sentences.lang == target_lang]

    bilang_sentences = source_sentences.merge(target_sentences, left_on='tran_id',
                                              right_on='id',
                                              suffixes=[source_lang, target_lang])
    bilang_sentences = bilang_sentences[['text' + source_lang, 'text' + target_lang]]

    # save results
    file_name = os.path.join(data_dir, '{source}-{target}.csv'.format(source=source_lang, target=target_lang))
    bilang_sentences.to_csv(file_name, index=False, sep='\t')
