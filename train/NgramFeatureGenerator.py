#encoding:utf-8

import numpy as np
from collections import Counter
import treetaggerwrapper
tagger = treetaggerwrapper.TreeTagger(TAGLANG='en',TAGDIR='/home/lr/hayashi/ra_web_app')
from nltk.tokenize import sent_tokenize

def ngram(words, n):
    return list(zip(*(words[i:] for i in range(n))))

def make_ngram_list(all_list, w_range):
    countlist  = []
    for i in all_list:
        out = ngram(i.split(), w_range)
        countlist.extend(out)

    with open("../dat/word." + str(w_range) +  "gram", "w") as fw:
        for n, (k,v) in enumerate(sorted(Counter(countlist).items(), key=lambda x:-x[1])):
            fw.write('%s\t%s\t%s\n' %  (n, k, v))

def make_pos_ngram_list(all_list, w_range):
    countlist  = []
    for i in all_list:
        out = ngram(i.split(), w_range)
        countlist.extend(out)

    with open("../dat/pos." + str(w_range) +  "gram", "w") as fw:
        for n, (k,v) in enumerate(sorted(Counter(countlist).items(), key=lambda x:-x[1])):
            fw.write('%s\t%s\t%s\n' %  (n, k, v))

def ngram_features(sentence, th1=1, th2=7, th3=7):
    uni_dic = {}
    bi_dic = {}
    tri_dic = {}
    with open('../dat/word.1gram', "r") as f1:
        for i in f1:
            if int(i.split('\t')[-1]) == th1:
                break
            else:
                uni_dic[str(i.split('\t')[1])] = int(i.split('\t')[0])

    with open('../dat/word.2gram', "r") as f2:
        for i in f2:
            if int(i.split('\t')[-1]) == th2:
                break
            else:
                bi_dic[str(i.split('\t')[1])] = int(i.split('\t')[0]) + len(uni_dic)

    with open('../dat/word.3gram', "r") as f3:
        for i in f3:
            if int(i.split('\t')[-1]) == th3:
                break
            else:
                tri_dic[str(i.split('\t')[1])] = int(i.split('\t')[0]) + len(uni_dic) + len(bi_dic)

    unigramlist = ngram(sentence.split(),1)
    bigramlist = ngram(sentence.split(),2)
    trigramlist = ngram(sentence.split(),3)
    out = []
    for i in unigramlist:
        if str(i) in uni_dic.keys():
            out.append(uni_dic[str(i)])

    for i in bigramlist:
        if str(i) in bi_dic.keys():
            out.append(bi_dic[str(i)])

    for i in trigramlist:
        if str(i) in tri_dic.keys():
            out.append(tri_dic[str(i)])

    out_array = np.zeros(len(uni_dic) + len(bi_dic) + len(tri_dic))
    for k, v in Counter(out).items():
        out_array[k] = v

    out_array = out_array.reshape(1, len(out_array))
    return out_array

def pngram_features(sentence, th1=1, th2=7, th3=7):
    uni_dic = {}
    bi_dic = {}
    tri_dic = {}
    with open('../dat/pos.1gram', "r") as f1:
        for i in f1:
            if int(i.split('\t')[-1]) == th1:
                break
            else:
                uni_dic[str(i.split('\t')[1])] = int(i.split('\t')[0])

    with open('../dat/pos.2gram', "r") as f2:
        for i in f2:
            if int(i.split('\t')[-1]) == th2:
                break
            else:
                bi_dic[str(i.split('\t')[1])] = int(i.split('\t')[0]) + len(uni_dic)

    with open('../dat/pos.3gram', "r") as f3:
        for i in f3:
            if int(i.split('\t')[-1]) == th3:
                break
            else:
                tri_dic[str(i.split('\t')[1])] = int(i.split('\t')[0]) + len(uni_dic) + len(bi_dic)

    unigramlist = ngram(sentence.split(),1)
    bigramlist = ngram(sentence.split(),2)
    trigramlist = ngram(sentence.split(),3)
    out = []
    for i in unigramlist:
        if str(i) in uni_dic.keys():
            out.append(uni_dic[str(i)])

    for i in bigramlist:
        if str(i) in bi_dic.keys():
            out.append(bi_dic[str(i)])

    for i in trigramlist:
        if str(i) in tri_dic.keys():
            out.append(tri_dic[str(i)])

    out_array = np.zeros(len(uni_dic) + len(bi_dic) + len(tri_dic))
    for k, v in Counter(out).items():
        out_array[k] = v

    out_array = out_array.reshape(1, len(out_array))
    return out_array

if __name__ == '__main__':
    out = []
    _tmp = []
    with open("/home/lr/hayashi/github/GradeSystemForEssay/cefrj/original_correct_pairs/all.raw", "r") as f:
        for i in f:
            o = sent_tokenize(i.rstrip().rstrip())
            tagged = [tagger.TagText(o)]
            for sentence in tagged:
                try:
                    for word in sentence:
                        _tmp.append(str(word.split('\t')[1]))
                    out.append(' '.join(_tmp))
                except:
                    pass
                _tmp = []

    #ngram_features(test1, 'G20_34_P1')
    #make_ngram_list(out, 1)
    #make_ngram_list(out, 2)
    #make_ngram_list(out, 3)
    make_pos_ngram_list(out, 1)
    make_pos_ngram_list(out, 2)
    make_pos_ngram_list(out, 3)
