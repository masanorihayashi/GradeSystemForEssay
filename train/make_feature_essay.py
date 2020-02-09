#encoding:utf-8
from __future__ import print_function

import os
import re
import argparse
import numpy as np
from sklearn.externals import joblib

import mord
import treetaggerwrapper

import spacy
nlp = spacy.load('en_core_web_sm')

import nltk
from nltk.tokenize import sent_tokenize
from collections import Counter
from collections import OrderedDict
from sklearn.externals import joblib

a1 = "/home/lr/hayashi/ra/4research/dat/word_diffculty/a1.word"
a2 = "/home/lr/hayashi/ra/4research/dat/word_diffculty/a2.word"
b1 = "/home/lr/hayashi/ra/4research/dat/word_diffculty/b1.word"
b2 = "/home/lr/hayashi/ra/4research/dat/word_diffculty/b2.word"
fun = "/home/lr/hayashi/ra/data/raw_pos/function.list"

a1_words = []
a2_words = []
b1_words = []
b2_words = []
fun_words = []
diff_words = []

with open(a1) as fa1, open(a2) as fa2, open(b1) as fb1, open(b2) as fb2, open(fun) as ffn:
    for a1w in fa1:
        a1_words.append(a1w.lower().split()[0])
        diff_words.append(a1w.lower().split()[0])
    for a2w in fa2:
        a2_words.append(a2w.lower().split()[0])
        diff_words.append(a2w.lower().split()[0])
    for b1w in fb1:
        b1_words.append(b1w.lower().split()[0])
        diff_words.append(b1w.lower().split()[0])
    for b2w in fb2:
        b2_words.append(b2w.lower().split()[0])
        diff_words.append(b2w.lower().split()[0])
    for funw in ffn:
        fun_words.append(funw.lower().split()[0])
        diff_words.append(funw.lower().split()[0])

class Surface:
    def __init__(self, text):
        self.text = text
        self.sentences = sent_tokenize(self.text.lower())
        #これを基に前処理を行う
        self.sen_length = len(self.sentences)
        #remove symbol
        self.rm_symbol_sentences = [re.sub("[!-/:-@[-`{-~]", "", sentence) for sentence in self.sentences]
        self.prop_sentences = [str(re.sub(r"([+-]?[0-9]+\.?[0-9]*)", "NUM", sentence)) for sentence in self.rm_symbol_sentences]
        self.prop_words = ' '.join(self.prop_sentences).split()
        self.total_words = len(self.prop_words)
        self.word_types = set(self.prop_words)

    def stats(self):
        return [self.sen_length, self.total_words, float(len(self.word_types)/float(self.total_words))]

    def ngram(self):
        all_ngram = []
        for num in [1, 2, 3]:
            _ngrams = [list(zip(*(sentence.split()[i:] for i in range(num)))) for sentence in self.prop_sentences]
            ngrams = [flat for inner in _ngrams for flat in inner]
            all_ngram.extend(set(ngrams))
            '''
            for k,v in sorted(Counter(ngrams).items(), key=lambda x: -x[1]):
                if v < 5:
                    pass
                else:
                    print(k, v, sep='\t')
            ''' 
        return Counter(all_ngram)

    #使わない
    '''
    def pos_ngram(self, n):
        _pos_ngrams = [nltk.pos_tag(nltk.word_tokenize(sentence)) for sentence in self.sentences]
        pos_ngrams = [flat for inner in _pos_ngrams for flat in inner]
        #print(pos_ngrams)
        return Counter(pos_ngrams)
    '''

    def word_difficulty(self):
        '''
        !!!要修正!!!
        '''
        a1_ratio = len(self.word_types & set(a1_words))/ float(self.total_words)
        a2_ratio = len(self.word_types & set(a2_words))/ float(self.total_words)
        b1_ratio = len(self.word_types & set(b1_words))/ float(self.total_words)
        b2_ratio = len(self.word_types & set(b2_words))/ float(self.total_words)
        fun_ratio = len(self.word_types & set(fun_words))/ float(self.total_words)

        return [a1_ratio, a2_ratio, b1_ratio, b2_ratio, fun_ratio]

    def features(self):
        ngrams = self.ngram()
        stats  = self.stats()
        diff = self.word_difficulty()

        return ngrams, stats, diff


"""
#文法項目の読み込み
grmlist = []
num_grm_dic = {}
with open('re_modify.txt', 'r') as f:
    for num, i in enumerate(f):
        grmlist.append(i.rstrip())
        num_grm_dic[num] = i.rstrip()
"""

class GrmItem:
    def __init__(self, text):
        tagger = treetaggerwrapper.TreeTagger(TAGLANG='en',TAGDIR='/home/lr/hayashi/ra_web_app')
        self.text = text
        #小文字にすると拾えない
        self.sentences = sent_tokenize(self.text)
        self.tagged = [tagger.TagText(sentence) for sentence in self.sentences]
        self.parsed = [' '.join(sentence).replace('\t', '_') for sentence in self.tagged]

    def detect(self, grmlist):
        grm_dic = {}
        for num, grm in enumerate(grmlist, 1):
            try:
                _grm_freq = [re.findall(grm, sentence) for sentence in self.parsed]
                grm_freq = [flat for inner in _grm_freq for flat in inner]
                grm_dic[num] = len(grm_freq)
            except:
                pass

        return grm_dic

    def pos_ngram(self):
        _tmp = []
        pos_list = []
        for sentence in self.tagged:
            #print(sentence)
            try:
                for word in sentence:
                    _tmp.append(str(word.split('\t')[1]))
                pos_list.append(' '.join(_tmp))
            except:
                pass
            _tmp = []

        all_pos_ngrams = []
        for num in [1, 2, 3]:
            _pos_ngrams = [list(zip(*(sentence.split()[i:] for i in range(num)))) for sentence in pos_list]
            pos_ngrams = [flat for inner in _pos_ngrams for flat in inner]
            all_pos_ngrams.extend(pos_ngrams)
            '''
            for k,v in sorted(Counter(pos_ngrams).items(), key=lambda x: -x[1]):
                if v < 5:
                    pass
                else:
                    print(k, v, sep='\t')
            '''

        return Counter(all_pos_ngrams)

    def features(self):
        grmitem = self.detect(grmlist)
        pos_ngram = self.pos_ngram()
        for k, v in grmitem.items():
            if v == 0:
                del(grmitem[k])

        return grmitem, pos_ngram


class Feature:
    def __init__(self, ngram={}, pos_ngram={}, grmitem={}, word_difficulty={}, stats={}):
        self.ngram = ngram
        self.pos_ngram = pos_ngram
        self.grmitem = grmitem
        self.word_difficulty = word_difficulty
        self.stats = stats
        self.word_dic = {}
        self.pos_dic = {}
        for line in open("dat/word.dat", "r"):
            self.word_dic[line.split('\t')[1]] = line.split('\t')[0]
        for line in open("dat/pos.dat", "r"):
            self.pos_dic[line.split('\t')[1]]  = line.split('\t')[0]

    def ngram2vec(self):
        fdic = OrderedDict()
        #word ngram
        for feature in self.ngram:
            if str(feature) in self.word_dic:
                fdic[int(self.word_dic[str(feature)]) - 1] = self.ngram[feature]/float(self.stats[1])
            else:
                pass

        #pos ngram
        for feature in self.pos_ngram:
            if str(feature) in self.pos_dic:
                fdic[int(self.pos_dic[str(feature)]) - 1  + len(self.word_dic)] = self.pos_ngram[feature]/float(self.stats[1])
            else:
                pass

        #grm item
        for key, value in self.grmitem.items():
            fdic[int(key) - 1 + len(self.pos_dic) + len(self.word_dic)] = value/float(self.stats[1])

        #word diff
        for number, feature in enumerate(self.word_difficulty, 0):
            #501 is length of grm item
            fdic[number + int(501) + len(self.pos_dic) + len(self.word_dic)] = feature


        return fdic

    def concat(self):
        ngrams = self.ngram2vec()
        #vec_size =  4 + int(501) + len(self.pos_dic) + len(self.word_dic)
        #このハードコード は一体... 
        vec_size = 5 + int(501) + len(self.pos_dic) + len(self.word_dic)
        inputs = np.zeros([1, vec_size])

        for k, v in ngrams.items():
            inputs[0, k] =v

        return inputs[0]

#todo
'''
素性化はほぼ完了？
素性のIDのマッピング
単語難易度の推定
予測
 '''

####実験場######
#データめっちゃ作る
def dataset():
    data = ''
    trains = []
    lss = []
    for i in open('trainfile.dat'):
        with open(i.rstrip(), 'r') as f:
            if 'a1' in i: lss.append(1)
            if 'a2' in i: lss.append(2)
            if 'b1' in i: lss.append(3)
            if 'b2' in i: lss.append(4)
            if 'c1' in i: lss.append(5)
            if 'c2' in i: lss.append(6)

            for i in f:
                data += i.rstrip() + ' '
            surface = Surface(unicode(data))
            ngram, stats, diff = surface.features()
            grmitem = GrmItem(unicode(data))
            grm, pos_ngram  = grmitem.features()
            inputs = Feature(ngram=ngram, pos_ngram=pos_ngram, grmitem=grm, word_difficulty=diff, stats=stats).concat()
            trains.append(inputs)
            data = ''
        print(len(trains))
        #if (len(trains)) == 10:
        #    return trains, np.array(lss)
    return trains, np.array(lss)

#素性作成→予測まで
def main():

    pass

def feature():
    path = "/home/lr/hayashi/github/GradeSystemForEssay/cefrj/original_correct_pairs/all.raw"
    all_sen = []
    with open(path, "r") as f:
        for i in f.readlines():
            all_sen.append(i.rstrip().lower())

    _ngrams = [list(zip(*(sentence.split()[i:] for i in range(3)))) for sentence in all_sen]
    ngrams = [flat for inner in _ngrams for flat in inner]
    for n, (k, v) in enumerate(sorted(Counter(ngrams).items(), key=lambda x:-x[1])):
         print('%s\t%s\t%s' %  (n, k, v))

def pos():
    tagger = treetaggerwrapper.TreeTagger(TAGLANG='en',TAGDIR='/home/lr/hayashi/ra_web_app')

    path = "/home/lr/hayashi/github/GradeSystemForEssay/cefrj/original_correct_pairs/all.raw"
    all_sen = []
    with open(path, "r") as f:
        for i in f.readlines():
            all_sen.append(i.rstrip().lower())
    tagged = [tagger.TagText(sentence) for sentence in all_sen]

    print(tagged)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input data')
    args = parser.parse_args()
    #main()
    #train()
    #main()
    #feature()
    pos()
