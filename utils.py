from corpus import contractions
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np


def remove_stopwords(sentence, customize_stopwords=None):
    stops = list(set(stopwords.words("english")))
    if customize_stopwords is not None:
        stops = stops + customize_stopwords
    sentence = [w for w in sentence.split(" ") if not w in stops]
    return " ".join(sentence)


def contract_split(sentence):
    sentence = " ".join([contractions[word] if word in contractions else word for word in sentence.lower().split(' ')])
    return sentence


def clean_text(text, not_removed_chars=[], only_charter=True, include_number=True):
    chars = "".join(not_removed_chars)
    """"
    "he'll" -> "he will"
    """
    text = contract_split(text)

    """"
    replace "?" - >" ? "
    """
    text = re.sub(fr"([?.!,{chars}])", r" \1 ", text)

    """"
    remove redundant space "  " -> ""
    """
    text = re.sub(r'[" "]+', " ", text)

    """"
    replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    """
    if only_charter and include_number:
        text = re.sub(fr"[^a-zA-Z0-9{chars}]+", " ", text)
    elif only_charter and not include_number:
        text = re.sub(fr"[^a-zA-Z{chars}]+", " ", text)
    elif not only_charter and include_number:
        text = re.sub(fr"[^a-zA-Z?.!,0-9{chars}]+", " ", text)
    elif not only_charter and not include_number:
        text = re.sub(fr"[^a-zA-Z?.!,{chars}]+", " ", text)

    text = text.replace("  ", " ")

    text = text.strip()

    return text


def tokenize(s, moses_tok=False):
    if moses_tok:
        s = ' '.join(word_tokenize(s))
        s = s.replace(" n't ", "n 't ")  # HACK to get ~MOSES tokenization
        return s.split()
    else:
        return word_tokenize(s)


def add_start_end_token(sentence, start="<start>", end="<end>"):
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    sentence = f'{start} ' + sentence + f' {end}'
    return sentence.strip()


def update_word2vec(words_set, word2vec_matrix, padding_value="eos"):
    for word in words_set:
        if word not in word2vec_matrix:
            word2vec_matrix[word] = np.random.uniform(low=-0.1, high=0.1, size=300)
    if padding_value not in word2vec_matrix:
        word2vec_matrix[padding_value] = np.random.uniform(low=-0.00001, high=0.00001, size=300)


def get_words_set(sentence_word_list):
    words_list = []
    for word_list in sentence_word_list:
        words_list.extend(word_list)
    return set(words_list)


def get_batch(batch, word_emb_dim, word2vec):
    embed = np.zeros((len(batch),
                      len(batch[0]),
                      word_emb_dim))
    for i in range(len(batch)):
        for j in range(len(batch[i])):
            embed[i, j, :] = word2vec[batch[i][j]]
    return embed


def word2vec_encode(sentences, word_emb_dim, word2vec, bath_size=64):
    embeddings = []
    for stidx in range(0, len(sentences), bath_size):
        batch = sentences[stidx:stidx + bath_size]
        batch = get_batch(batch, word_emb_dim, word2vec)
        embeddings.append(batch)
    embeddings = np.vstack(embeddings)
    return embeddings


def text_preprocess_pipeline(text, func_list=[clean_text, remove_stopwords, tokenize]):
    if len(func_list) == 0:
        return text
    return text_preprocess_pipeline(func_list[0](text), func_list[1:])
