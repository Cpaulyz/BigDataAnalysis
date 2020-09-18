import os
import re
from math import log

from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize

all_dic = {}
root_path = '..\\resources\\ACL2020'
invalid_word = stopwords.words('english') + ['geca','r-qae','bias','wiqa']


# 获取单词的词性
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def add_to_dict(word_list, dic):
    for word in word_list:
        word = str(word).lower()
        if is_valid(word):
            if word in dic.keys():
                dic[word] += 1
            else:
                dic[word] = 1


# 预处理,如果是False就丢掉
def is_valid(word):
    if re.match("[()\-:;,.0-9]+", word) or word in invalid_word:
        return False
    elif len(word) < 4:
        return False
    else:
        return True


def read(path):
    str = ''
    local_dic = {}
    with open(root_path + "\\" + path, "r", encoding='UTF-8') as f:  # 设置文件对象
        lines = f.readlines()  # 可以是随便对文件的操作
        ready = False
        for line in lines:
            line = line.strip()
            if line == "References":
                print('end read', path)
                break
            elif line == "Abstract":
                print('start read', path)
                ready = True
            elif ready:
                if line == '':
                    continue
                elif line[-1] == '-':
                    str += line[:-1]
                else:
                    str += line
        print(str)
    sens = sent_tokenize(str)
    for sentence in sens:
        # print(sentence)
        tokens = word_tokenize(sentence)  # 分词
        tagged_sent = pos_tag(tokens)  # 获取单词词性

        wnl = WordNetLemmatizer()
        lemmas_sent = []
        for tag in tagged_sent:
            wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))  # 词形还原
        # print(lemmas_sent)
        add_to_dict(lemmas_sent, local_dic)
    all_dic[path] = local_dic


def TFIDF():
    article_nums = len(all_dic)
    for article in all_dic.keys():
        article_dict: dict = all_dic[article]
        article_word_counts = 0
        for count in article_dict.values():
            article_word_counts += count
        local_dict = {}
        for word in article_dict:
            TF = article_dict[word] / article_word_counts
            contain_count = 1  # 包含的文档总数，因为要+1，干脆直接初始值为1来做
            for article1 in all_dic.keys():
                if word in all_dic[article1].keys():
                    contain_count += 1
            IDF = log(article_nums / contain_count)
            local_dict[word] = TF * IDF
        all_dic[article] = local_dict  # 用TFIDF替代词频


if __name__ == '__main__':
    files_name = os.listdir(root_path)
    # num = 0
    for file_name in files_name:
        if file_name.endswith(".txt"):
            print(file_name)
            read(file_name)
            # num += 1
            # if num > 2:
            #     break
    TFIDF()
    # print(all_dic)
    with open("method1_dict.txt", 'w', encoding="UTF-8") as f:
        for dic in all_dic.values():
            tmp = sorted(dic.items(), key=lambda x: x[1], reverse=True)
            # print(a)
            f.write(tmp[0][0] + ' ' + str(tmp[0][1]) + '\n')
