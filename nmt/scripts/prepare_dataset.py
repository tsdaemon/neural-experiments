import argparse
import os
import shutil

import nltk
import pandas as pd
from tokenize_uk import tokenize_words as tokenizer_ukr
from tqdm import tqdm
import torch

from nmt.containers.vocab import Vocab
from nmt.Constants import *

parser = argparse.ArgumentParser()
parser.add_argument('-s', help='ISO 639-3 code for a source language', required=True)
parser.add_argument('-t', help='ISO 639-3 code for a target language', required=True)
parser.add_argument('-max_sentence_length', default=8)
parser.add_argument('-min_word_count', default=4, help='Minimum number of word occurrence '
                                                       'in corpora')
parser.add_argument('-data_dir', default='./data', help='Data directory')

tokenizer_deu = nltk.tokenize.WordPunctTokenizer().tokenize

tokenizers = {
    'ukr': tokenizer_ukr,
    'deu': tokenizer_deu
}


def preprocess_corpora(sents, tokenizer, min_word_count):
    n_words = {}

    sents_tokenized = []
    for sent in tqdm(sents, desc='Reading corpora'):
        sent_tokenized = tokenizer(sent)

        sents_tokenized.append(sent_tokenized)

        for word in sents_tokenized:
            if word in n_words:
                n_words[word] += 1
            else:
                n_words[word] = 1

    for i, sent_tokenized in enumerate(sents_tokenized):
        sent_tokenized = [t if n_words[t] >= min_word_count else UNK_token for t in sent_tokenized]
        sents_tokenized[i] = sent_tokenized

    return sents_tokenized


def read_vocab(sents):
    vocab = Vocab()
    for sent in sents:
        vocab.index_words(sent)

    return vocab


def indexes_from_sentence(vocab, sentence):
    return [vocab.word2index[word] for word in sentence]


def tensor_from_sentence(lang, sentence, max_seq_length):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    indexes.insert(0, SOS_token)
    # we need to have all sequences the same length to process them in batches
    if len(indexes) < max_seq_length:
        indexes += [PAD_token] * (max_seq_length - len(indexes))
    tensor = torch.LongTensor(indexes)
    return tensor


def tensors_from_pair(source_sent, target_sent, max_seq_length):
    source_tensor = tensor_from_sentence(source_vocab, source_sent, max_seq_length).unsqueeze(1)
    target_tensor = tensor_from_sentence(target_vocab, target_sent, max_seq_length).unsqueeze(1)

    return (source_tensor, target_tensor)


if __name__ == '__main__':
    args = parser.parse_args()
    source_lang = args.s
    target_lang = args.t

    data_dir = args.data_dir
    lang_dir = os.path.join(data_dir, '{}_{}'.format(source_lang, target_lang))
    if os.path.isdir(lang_dir):
        shutil.rmtree(lang_dir)
    os.mkdir(lang_dir)

    max_length = args.max_sentence_length
    min_word_count = args.min_word_count

    input_file = os.path.join(data_dir, '{source}-{target}.csv'.format(source=source_lang,
                                                                       target=target_lang))
    corpora = pd.read_csv(input_file, delimiter='\t')
    source_corpora = preprocess_corpora(corpora['text' + source_lang], tokenizers[source_lang], min_word_count)
    target_corpora = preprocess_corpora(corpora['text' + target_lang], tokenizers[target_lang], min_word_count)

    source_corpora, target_corpora = zip(
        *[(s, t) for s, t in zip(source_corpora, target_corpora)
          if len(s) < max_length and len(t) < max_length]
    )

    source_vocab = read_vocab(source_corpora)
    target_vocab = read_vocab(target_corpora)
    target_vocab.to_file(os.path.join(lang_dir, '{}.vocab.txt'.format(target_lang)))
    source_vocab.to_file(os.path.join(lang_dir, '{}.vocab.txt'.format(source_lang)))

    max_seq_length = max_length + 2  # 2 for EOS_token and SOS_token

    tensors = []
    for source_sent, target_sent in zip(source_corpora, target_corpora):
        tensors.append(tensors_from_pair(source_sent, target_sent, max_seq_length))

    x, y = zip(*tensors)
    x = torch.transpose(torch.cat(x, dim=-1), 1, 0)
    y = torch.transpose(torch.cat(y, dim=-1), 1, 0)
    torch.save(x, os.path.join(lang_dir, 'x.bin'))
    torch.save(y, os.path.join(lang_dir, 'y.bin'))
