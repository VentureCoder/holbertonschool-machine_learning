#!/usr/bin/env python3
"""Dataset class that loads and preps a dataset for machine translation"""
import tensorflow as tf
import transformers
from setup import load_pt2en


class Dataset:
    """Loads and preps a dataset for machine translation"""

    def __init__(self):
        """Initializes the dataset, tokenizers, and encodes the data"""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """Creates sub-word tokenizers for the dataset

        Args:
            data: tf.data.Dataset whose examples are tuples (pt, en)

        Returns:
            tokenizer_pt, tokenizer_en: the trained tokenizers
        """
        pt_sentences = []
        en_sentences = []

        for pt, en in data:
            pt_sentences.append(pt.numpy().decode('utf-8'))
            en_sentences.append(en.numpy().decode('utf-8'))

        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_sentences, vocab_size=2 ** 13
        )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_sentences, vocab_size=2 ** 13
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Encodes a translation pair into tokens

        Args:
            pt: tf.Tensor containing the Portuguese sentence
            en: tf.Tensor containing the corresponding English sentence

        Returns:
            pt_tokens, en_tokens: lists of tokens with start and end tokens
        """
        pt_text = pt.numpy().decode('utf-8')
        en_text = en.numpy().decode('utf-8')

        vocab_pt = self.tokenizer_pt.vocab_size
        vocab_en = self.tokenizer_en.vocab_size

        pt_tokens = self.tokenizer_pt.encode(
            pt_text, add_special_tokens=False
        )
        en_tokens = self.tokenizer_en.encode(
            en_text, add_special_tokens=False
        )

        pt_tokens = [vocab_pt] + pt_tokens + [vocab_pt + 1]
        en_tokens = [vocab_en] + en_tokens + [vocab_en + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """TensorFlow wrapper for the encode instance method

        Args:
            pt: tf.Tensor containing the Portuguese sentence
            en: tf.Tensor containing the corresponding English sentence

        Returns:
            pt_tokens, en_tokens: tf.Tensors with shape set
        """
        pt_tokens, en_tokens = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )

        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens
