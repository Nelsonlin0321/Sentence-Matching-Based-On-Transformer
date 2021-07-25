import tensorflow as tf
import numpy as np
import corpus
import utils


class TextPreprocessor(object):

    def __init__(self, pipeline=None, padding_config=None, token2id=True, word2vec_path=None):
        """
        :param pipeline:  list of functions to preprocess list of text
        :param padding_config: padding config: default =
        {'padding': 'post',
        'truncating': 'post',
        'padding value': 0}
        :param token2id:  Ture of False to convert word to token id
        """

        if pipeline is None:
            pipeline = []

        self.tokenizer_config = {"num_words": None,
                                 'filters': '',
                                 "lower": True,
                                 'split': ' ',
                                 'char_level': False,
                                 'oov_token': None,
                                 'document_count': 0,
                                 }

        self.default_padding_config = {'padding': 'post',
                                       'truncating': 'post',
                                       'padding value': 0}
        self.word2vec_path = word2vec_path
        self.word2vec = None
        self.word_emb_dim = None

        self.token2id = token2id
        self.pipeline = pipeline
        self.tokenizer = None
        self.padding_config = self.default_padding_config

        # update
        if padding_config is not None:
            for parameter in padding_config:
                if parameter in self.padding_config:
                    self.padding_config[parameter] = padding_config[parameter]

        if self.token2id:
            self.tokenizer = tf.keras.preprocessing.text.Tokenizer(self.tokenizer_config)

    def transform(self, text_list):
        # preprocess pipeline
        text_list = [self.__pipeline_transform(text, self.pipeline) for text in text_list]

        # covert token to ids
        if self.token2id:
            self.tokenizer.fit_on_texts(text_list)
            """tokenize a sentence to be a list of token id"""
            token_ids = self.tokenizer.texts_to_sequences(text_list)
            token_ids_padding = tf.keras.preprocessing.sequence.pad_sequences(token_ids,
                                                                              padding=self.padding_config["padding"],
                                                                              truncating=self.padding_config[
                                                                                  "truncating"],
                                                                              value=self.padding_config[
                                                                                  "padding value"],
                                                                              dtype='int32')
            return token_ids_padding
        else:
            tokens_padding = tf.keras.preprocessing.sequence.pad_sequences(text_list,
                                                                           padding=self.padding_config["padding"],
                                                                           truncating=self.padding_config[
                                                                               "truncating"],
                                                                           value=self.padding_config["padding value"],
                                                                           dtype=np.object)

            if self.word2vec_path is not None:
                if self.word2vec is None:
                    self.word2vec = corpus.read_word2vec(self.word2vec_path)
                words_set = utils.get_words_set(tokens_padding)
                utils.update_word2vec(words_set, self.word2vec, "eos")
                self.word_emb_dim = len(self.word2vec["eos"])
                tokens_word_embed = utils.word2vec_encode(tokens_padding, self.word_emb_dim, self.word2vec,
                                                          bath_size=64)

                return tokens_word_embed

            return tokens_padding

    def __pipeline_transform(self, text, pipeline):
        if len(pipeline) == 0:
            return text
        return self.__pipeline_transform(pipeline[0](text), pipeline[1:])


if __name__ == "__main__":
    from utils import clean_text, remove_stopwords, tokenize

    sentence = u"¿Puedo tomar prestado este libro?"
    print(clean_text(sentence, unremove_chars=["¿"]))
