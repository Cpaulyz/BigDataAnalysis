import os
import re

from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize

words = {}  # 存放的数据格式为(Key:String,relative_word:Array)
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


def add_to_dict(word_list, windows=5):
    valid_word_list = []  # 先进行过滤

    for word in word_list:
        word = str(word).lower()
        if is_valid(word):
            valid_word_list.append(word)

    # 根据窗口进行关系建立
    if len(valid_word_list) < windows:
        win = valid_word_list
        build_words_from_windows(win)
    else:
        index = 0
        while index + windows <= len(valid_word_list):
            win = valid_word_list[index:index + windows]
            index += 1
            build_words_from_windows(win)


# 根据小窗口，将关系建立到words中
def build_words_from_windows(win):
    for word in win:
        if word not in words.keys():
            words[word] = []
        for other in win:
            if other == word or other in words[word]:
                continue
            else:
                words[word].append(other)


# 预处理,如果是False就丢掉
def is_valid(word):
    if re.match("[()\-:;,.0-9]+", word) or word in invalid_word:
        return False
    elif len(word) < 4:
        return False
    else:
        return True


def text_rank(d=0.85, max_iter=100):
    min_diff = 0.05
    words_weight = {}  # {str,float)
    for word in words.keys():
        words_weight[word] = 1 / len(words.keys())
    for i in range(max_iter):
        n_words_weight = {}  # {str,float)
        max_diff = 0
        for word in words.keys():
            n_words_weight[word] = 1 - d
            for other in words[word]:
                if other == word or len(words[other]) == 0:
                    continue
                n_words_weight[word] += d * words_weight[other] / len(words[other])
            max_diff = max(n_words_weight[word] - words_weight[word], max_diff)
        words_weight = n_words_weight
        print('iter', i, 'max diff is', max_diff)
        if max_diff < min_diff:
            print('break with iter', i)
            break
    return words_weight


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
        add_to_dict(lemmas_sent, 5)


def start(topN=30):
    files_name = os.listdir(root_path)
    # num = 0
    for file_name in files_name:
        if file_name.endswith(".txt"):
            print(file_name)
            read(file_name)
            # num += 1
            # if num > 2:
            #     break
    words_weight = text_rank()
    tmp = sorted(words_weight.items(), key=lambda x: x[1], reverse=True)
    with open("method3_dict.txt", 'w', encoding="UTF-8") as f:
        for i in range(topN):
            f.write(tmp[i][0] + ' ' + str(tmp[i][1]) + '\n')
            print(tmp[i])
    print(words_weight)


if __name__ == '__main__':
    start()
