# 预处理完将分词文件存入words.txt文件
import os
import re

from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize

dic = {}
root_path = '..\\resources\\ACL2020'
invalid_word = stopwords.words('english')


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


def add_to_dict(word_list):
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
        # print(str)
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
        add_to_dict(lemmas_sent)


def start(topN=30):
    files_name = os.listdir(root_path)
    # num = 0
    for file_name in files_name:
        if file_name.endswith(".txt"):
            read(file_name)
            print('----------end', file_name, '----------------')
            # num += 1
            # if num > 2:
            #     break
    a = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    # print(a)
    with open("method2_dict.txt", 'w', encoding="UTF-8") as f:
        index = 0
        for word in a:
            f.write(word[0] + ' ' + str(word[1]) + '\n')
            index += 1
            if index >= topN:
                break
    # print(dic)


if __name__ == '__main__':
    start()
