import xml.etree.ElementTree as ET
import geniatagger as gt
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import os
import pickle
import gc
import numpy as np
import nltk
import sklearn
from sklearn.model_selection import cross_val_score
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

tagger_path = "/home/lightspace/PycharmProjects/npl/geniatagger-3.0.2/geniatagger"

tagger = gt.GeniaTagger(tagger_path, ["-nt"])


def to_collection(file_path, category):
    """ read the data from xml file that made by MAE and transform it into two data set,
    which represent the word set and the categories set respectively.

    args:
        file_path: string, the xml file path for reading the document and tag information
        category: string, the categories  tag that choose for building classifier.

    return:
        X: the set of each word
        Y: the set of IOB style tag for each word. I represent inside the entity, O: outside the entity, B; In the begain of entity

    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    ranges = []
    # 1. find the tag range
    for child in root.iter(category):
        ranges.append(child.attrib)

    # set the tag for each range in string
    text = root.find("./TEXT").text
    text_list = list(text)
    for tag in ranges:
        start = int(tag['start'])
        end = int(tag['end'])
        text_list[start] = '_SEK_' + tag['TYPE'] + "_" + text_list[start]
        text_list[end - 1] = text_list[end - 1]+'_ESK_'
    tokenized_sen = []
    tags = []
    for sentence in "".join(text_list).splitlines():
        if len(sentence) > 0:
            temp =sent_tokenize(sentence.strip())
            for s in temp:
                if len(s) < 400 :
                    words_temp = [i for i in word_tokenize(s) if i.strip() != '']
                    if len(words_temp) > 0:
                        tokenized_sen.append(words_temp)
    #print(tokenized_sen)
    # construct tag set
    for i, sentence in enumerate(tokenized_sen):
        has_tag = False
        tag_type = ""
        tag = []
        for j, word in enumerate(sentence):
            if len(word) == 0:
                continue
            if re.search(r'^_SEK_', word):
                temp = word.split('_')
                tag.append('B_'+temp[2])
                if re.search(r'_ESK_$', word) is None:
                    tag_type = temp[2]
                    has_tag = True
                tokenized_sen[i][j] = re.sub('^_SEK_'+temp[2]+"_" + '|' + '_ESK_$' ,'',word)

                continue

            if has_tag:
                tag.append("I_"+tag_type)
            else:
                tag.append("O")

            if re.search(r'_ESK_$', word):
                has_tag = False
                # if re.sub(r'_ESK_$','', word) == "":
                #     print(sentence)
                tokenized_sen[i][j] = re.sub(r'_ESK_$','', word)
        tags.append(tag)
    # print(tokenized_sen)
    # print(tags)
    return tokenized_sen, tags


def get_token_tags(tokenized_sentence):
    tag_sen = []
    for sen in tokenized_sentence:
        temp = " ".join(sen)
        temp2 = tagger.parse(temp)
        if len(temp2) != len(sen):
            print(temp2)
            print(temp)
        tag_sen.append(temp2)
    return tag_sen


re_INIT_CAP = r'^[A-Z][^A-Z]+'
re_ALL_CAPS = r'^[A-Z]+$'
re_CAPS_MIX = r'(?=.*[A-Z].*)(?=[a-z]).*'
re_HAS_PUNCT = r'-|/|:|\.'
re_HAS_DIGIT = r'[0-9]'
re_ALL_DIGIT = r'^[0-9]+$'
re_DIGIT_PUNCTUATION = r'(?=.*[0-9].*)(?=.*[-/:\.].*)^[0-9-/:\.]+$'
re_ALPHA_NUM = r'(?=.*[0-9].*)(?=.*[a-zA-Z].*)^[0-9A-Za-z]+$'
re_DATE1 = r'^\d{4}-(0[1-9]|1[0-2])-([0-2][0-9]|3[0-1])$'
re_DATE2 = r'^\d{4}$'
re_DATE3 = r'^([0-9]|1[0-2]|0[0-9])-([0-9]|[1-2][0-9]|3[0-1]|0[0-9])-(\d{4}|\d{2})$'
re_DATE4 = r'^([0-9]|1[0-2]|0[0-9])/([0-9]|[1-2][0-9]|3[0-1]|0[0-9])/(\d{4}|\d{2})$'
re_DATE5 = r'^([0-9]|1[0-2])/\'\d{2}$'
re_DATE6 = r'^\'\d{2}$'
re_DATE7 = r'^(\d{4}|\d{2})\'s$'
re_DATE8 = r'^(\d{4}|\d{2})s$'
re_DATE9 = r'^([0-9]|1[0-2]|0[0-9])\.([0-9]|[1-2][0-9]|3[0-1]|0[0-9])\.(\d{2})$'
re_DATE10 = r'^([0-9]|[1-2][0-9]|3[0-1]|0[0-9])([ A-Za-z]+)(\d{4}|\d{2})$'
re_DATE11 = r'^([ A-Za-z]+)(\d{4}|\d{2})$'
re_DATE12 = r'^([0-9]|1[0-2]|0[0-9])/(\d{2,4})$'
re_DATE13 = r'^((Monday){1}|(Sunday){1}|(Tuesday){1}|(Wednesday){1}|(Thursday){1}|(Friday){1}|(Saturday){1})s?$'
re_DATE14 = r'^(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)$'
re_month = "(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
re_DATE15 = r'^(\d{2}-' + re_month + r'-\d{4})$'
re_USERNAME = r'^[A-Za-z]{2,3}\d{1,3}$'
re_AGE1 = r'^\d+[A-Za-z]+$'
re_is_1_2_digit = r'^\d{2}$'
re_is_comma = r'^[,]$'
re_DATE16 = r'^\d{1}/\d{1}$'
re_DATE17 = r'^((3rd)|(\dth))$'
def word_to_features(sent, i):
    # print(sent)
    # print(i)
    word = sent[i][0]
    base_form = sent[i][1]
    pos_tag = sent[i][2]
    chunk_tag = sent[i][3]
    #name_entity = sent[i][4]

    # token features
    features = {
        'bias': 1.0,
        'base_form': base_form,
        'pos_tag': pos_tag,
        'chunk_tag': chunk_tag,
    }
    # Contextual Features
    for j in range(1, 4):
        if i - j >= 0:
            temp_word = sent[i - j][0]
            temp_base_form = sent[i - j][1]
            temp_pos_tag = sent[i - j][2]
            temp_chunk_tag = sent[i - j][3]
            #temp_name_entity = sent[i - j][4]
            if re.search(re_DATE14, temp_word):
                temp_is_month = True
            else:
                temp_is_month = False

            if re.search(re_is_comma, temp_word):
                temp_is_comma = True
            else:
                temp_is_comma = False

            if re.search(re_is_1_2_digit, temp_word):
                temp_is_1_2_digit = True
            else:
                temp_is_1_2_digit = False

            if re.search(re_DATE2, temp_word):
                temp_is_4_digit = True
            else:
                temp_is_4_digit = False


            features.update({
                '-' + str(j) + ':base_form': temp_base_form,
                '-' + str(j) + ':pos_tag': temp_pos_tag,
                '-' + str(j) + ':chunk_tag': temp_chunk_tag,
                '-' + str(j) + 'is_month': temp_is_month,
                '-' + str(j) + 'is_comma': temp_is_comma,
                '-' + str(j) + 'is_1_2_digit': temp_is_1_2_digit,
                '-' + str(j) + 'is_4_digit': temp_is_4_digit

            })
        else:
            features.update({
                '-' + str(j) + ':base_form': "<S>",
                '-' + str(j) + ':pos_tag': "<S>",
                '-' + str(j) + ':chunk_tag': "<S>"
            })
        if i + j < len(sent):
            temp_word = sent[i + j][0]
            temp_base_form = sent[i + j][1]
            temp_pos_tag = sent[i + j][2]
            temp_chunk_tag = sent[i + j][3]
            #temp_name_entity = sent[i + j][4]
            if re.search(re_DATE14, temp_word):
                temp_is_month = True
            else:
                temp_is_month = False

            if re.search(re_is_comma, temp_word):
                temp_is_comma = True
            else:
                temp_is_comma = False

            if re.search(re_is_1_2_digit, temp_word):
                temp_is_1_2_digit = True
            else:
                temp_is_1_2_digit = False

            if re.search(re_DATE2, temp_word):
                temp_is_4_digit = True
            else:
                temp_is_4_digit = False

            features.update({
                '+' + str(j) + ':base_form': temp_base_form,
                '+' + str(j) + ':pos_tag': temp_pos_tag,
                '+' + str(j) + ':chunk_tag': temp_chunk_tag,
                '+' + str(j) + 'is_month':temp_is_month,
                '+' + str(j) + 'is_comma':temp_is_comma,
                '+' + str(j) + 'is_1_2_digit':temp_is_1_2_digit,
                '+' + str(j) + 'is_4_digit': temp_is_4_digit

            })
        else:
            features.update({
                '+' + str(j) + ':base_form': "</S>",
                '+' + str(j) + ':pos_tag': "</S>",
                '+' + str(j) + ':chunk_tag': "</S>"
            })
    # Orthographic Features
    if re.search(re_ALL_CAPS, word):
        features['all_caps'] = True
    if re.search(re_CAPS_MIX, word):
        features['caps_mix'] = True
    if re.search(re_INIT_CAP, word):
        features['init_cap'] = True
    if re.search(re_ALL_DIGIT, word):
        features['all_digit'] = True
    if re.search(re_ALPHA_NUM, word):
        features['alpha_num'] = True
    if re.search(re_HAS_DIGIT, word):
        features['has_digit'] = True
    if re.search(re_DIGIT_PUNCTUATION, word):
        features['digit_punct'] = True
    if re.search(re_HAS_PUNCT, word):
        features['has_punct'] = True
    if re.search(re_is_comma, word):
        features['is_comma'] = True
    if re.search(re_is_1_2_digit, word ):
        features["is_1_2_digit"] = True


    for j in range(1,18):
        a = "re_DATE"+str(j)
        if re.search(eval(a), word):
            features[eval(a)] = True
    return features


def sent2features(sent):
    return [word_to_features(sent, i) for i in range(len(sent))]


def read_files(path, category, n):
    files = os.listdir(path)

    paths = []
    for file in files:
        if not os.path.isdir(os.path.join(os.path.abspath(path), file)):
            paths.append(os.path.join(path, file))
    for i in range(0, len(paths), n):

        x = []
        y = []
        for j in range(i, min(n+i, len(paths))):
            print(files[j])
            one_x, one_y = to_collection(os.path.join(path, files[j]), category)
            n_one_x = get_token_tags(one_x)
            one_doc_feature = []
            for index, sent in enumerate(n_one_x):
                temp = sent2features(sent)
                if len(temp) != len(one_y[index]):
                    one_doc_feature = None
                    break
                else:
                    one_doc_feature.append(temp)
            if one_doc_feature is not None:
                x.extend(one_doc_feature)
                y.extend(one_y)

        yield x, y


def evalutation(model, x, y):
    labels = list(model.classes_)
    labels.remove('O')
    y_pre = model.predict(x)
    # for i in range(len(y_pre)):
    #     if len(y_pre[i]) != len(y[i]):
    #         print(i)
    #         print(len(x[i]))
    #         print(len(y[i]))
    #         print(len(y_pre[i]))
    print("The total f1-score")
    print(metrics.flat_f1_score(y, y_pre, average='weighted',labels = labels, zero_division=1))
    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )
    print(metrics.flat_classification_report(
        y, y_pre, labels=sorted_labels, digits=3, zero_division=1
    ))


def build_model_test(category):
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1 = 0.1,
        c2 = 0.1,
        max_iterations= 200,
        all_possible_transitions= True
    )
    iterator = read_files("/home/lightspace/Documents/course/npl/project/training-PHI-Gold-Set1", category, 1000)
    for x, y in iterator:
        crf.fit(x, y)
    iterator = read_files("/home/lightspace/Documents/course/npl/project/training-PHI-Gold-Set2", category, 1000)
    for x, y in iterator:
        crf.fit(x, y)
    file = open(os.path.join(".", category),"wb")
    pickle.dump(crf, file)
    file.close()

    file = open(os.path.join(".", category), "rb")
    model = pickle.load(file)

    test_set = read_files("/home/lightspace/PycharmProjects/npl/TEST_SET", category, 1)
    for x_test, y_test in test_set:
        # print(x_test)
        # print(y_test)
        print("result for " + category)
        evalutation(model, x_test, y_test)


if __name__ == "__main__":
    build_model_test("DATE")


