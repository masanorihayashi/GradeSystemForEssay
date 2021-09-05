#encoding:utf-8

from __future__ import print_function

import os
import re
import random
import pickle
import argparse
import numpy as np
from sklearn.externals import joblib

import mord
import treetaggerwrapper
from bs4 import BeautifulSoup as bs


import nltk
from nltk.tokenize import sent_tokenize
from nltk import word_tokenize as wt
from nltk import pos_tag
from collections import Counter
from collections import OrderedDict
from sklearn.externals import joblib

a1 = "../dat/a1.word"
a2 = "../dat/a2.word"
b1 = "../dat/b1.word"
fun = "../dat/func.word"

a1_words = []
a2_words = []
b1_words = []
fun_words = []
diff_words = []

with open(a1) as fa1, open(a2) as fa2, open(b1) as fb1, open(fun) as ffn:
    for a1w in fa1:
        a1_words.append(a1w.lower().split()[0])
        diff_words.append(a1w.lower().split()[0])
    for a2w in fa2:
        a2_words.append(a2w.lower().split()[0])
        diff_words.append(a2w.lower().split()[0])
    for b1w in fb1:
        b1_words.append(b1w.lower().split()[0])
        diff_words.append(b1w.lower().split()[0])
    for funw in ffn:
        fun_words.append(funw.lower().split()[0])
        diff_words.append(funw.lower().split()[0])

#機能語読み込み
function_dic = {}
func = open("../dat/func.word","r")
for num, i in enumerate(func.readlines()):
    function_dic[str(i.rstrip())] = str(num+1)

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
        xx = 1
        #for num in [1, 2, 3]:
        for num in [1, 2]:
            _ngrams = [list(zip(*(sentence.split()[i:] for i in range(num)))) for sentence in self.prop_sentences]
            ngrams = [flat for inner in _ngrams for flat in inner]
            all_ngram.extend(set(ngrams))
            '''
            for k,v in sorted(Counter(ngrams).items(), key=lambda x: -x[1]):
                if v < 5:
                    pass
                else:
                    print(xx, k, v, sep='\t')
                    xx += 1
            ''' 
        return Counter(all_ngram)

    def word_difficulty(self):
        '''
        !!!要修正!!!
        '''
        a1_ratio = len(self.word_types & set(a1_words))/ float(self.total_words)
        a2_ratio = len(self.word_types & set(a2_words))/ float(self.total_words)
        b1_ratio = len(self.word_types & set(b1_words))/ float(self.total_words)
        fun_ratio = len(self.word_types & set(fun_words))/ float(self.total_words)

        return [a1_ratio, a2_ratio, b1_ratio, fun_ratio]

    def features(self):
        ngrams = self.ngram()
        stats  = self.stats()
        diff = self.word_difficulty()

        return ngrams, stats, diff


#文法項目の読み込み
grmlist = []
num_grm_dic = {}
num_list_dic = {}
with open('../dat/grmitem.txt', 'r') as f:
    for num, i in enumerate(f, 1):
        grmlist.append(i.rstrip().split('\t')[1])
        num_grm_dic[num] = i.rstrip().split('\t')[1]
        num_list_dic[num] = i.rstrip().split('\t')[0]

class GrmItem:
    def __init__(self, text):
        tagger = treetaggerwrapper.TreeTagger(TAGLANG='en',TAGDIR='/home/lr/hayashi/ra_web_app')
        self.text = text
        #小文字にすると拾えない
        self.sentences = sent_tokenize(self.text)
        self.tagged = [tagger.TagText(sentence) for sentence in self.sentences]
        self.parsed = [' '.join(sentence).replace('\t', '_') for sentence in self.tagged]

    def detect(self, grmlist, itemslist):
        grm_dic = {}
        use_item = []
        for num, grm in enumerate(grmlist, 1):
            try:
                _grm_freq = [re.findall(grm, sentence) for sentence in self.parsed]
                grm_freq = [flat for inner in _grm_freq for flat in inner]
                if len(grm_freq) != 0:
                    grm_dic[num] = len(grm_freq)
                    use_item.append(itemslist[num])
            except:
                pass

        return grm_dic, use_item

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
        xx = 1
        #for num in [1, 2, 3]:
        for num in [1, 2]:
            _pos_ngrams = [list(zip(*(sentence.split()[i:] for i in range(num)))) for sentence in pos_list]
            pos_ngrams = [flat for inner in _pos_ngrams for flat in inner]
            all_pos_ngrams.extend(pos_ngrams)
            '''

            for k,v in sorted(Counter(pos_ngrams).items(), key=lambda x: -x[1]):
                if v < 5:
                    pass
                else:
                    print(xx, k, v, sep='\t')
                    xx += 1
            '''

        return Counter(all_pos_ngrams)

    def features(self):
        grmitem, use_list = self.detect(grmlist, num_list_dic)
        pos_ngram = self.pos_ngram()
        for k, v in grmitem.items():
            if v == 0:
                del(grmitem[k])

        return grmitem, pos_ngram, use_list


class Feature:
    def __init__(self, ngram={}, pos_ngram={}, grmitem={}, word_difficulty={}, stats={}):
        self.ngram = ngram
        self.pos_ngram = pos_ngram
        self.grmitem = grmitem
        self.word_difficulty = word_difficulty
        self.stats = stats
        self.word_dic = {}
        self.pos_dic = {}
        for line in open("../dat/word_essay.dat", "r"):
            self.word_dic[line.split('\t')[1]] = line.split('\t')[0]
        for line in open("../dat/pos_essay.dat", "r"):
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
        vec_size =   4 + int(501) + len(self.pos_dic) + len(self.word_dic)
        inputs = np.zeros([1, vec_size])

        for k, v in ngrams.items():
            inputs[0, k] =v

        return inputs[0]

def output(grade, stats, word_diff, grmitem):
    grade_class = {1:'A1', 2:'A2', 3:'B1', 4:'B2', 5:'C1'}
    output_dic = {}
    output_dic['grade'] = grade_class[grade[0]]
    output_dic['stats'] = stats
    output_dic['word_diff'] = word_diff
    output_dic['grmitem'] = grmitem

    return output_dic

#xmlデータから入力，出力，アライメント情報を抽出
def extract_dp_sentence(xml):
    dp_sentence = []
    ori_sentence = []
    correct_sentence = []
    tmp_line = ""
    tmp_corrected = ""
    for line_ in xml:
        line = line_.rstrip()
        if tmp_line == '<trial no="01a">':
            dp_sentence.append(line)
            tmp_line = line.rstrip()
        elif tmp_line == '<sentence psn="ns">':
            ori_sentence.append(line)
            tmp_line = line.rstrip()
        elif tmp_line == '<sentence psn="st">':
            correct_sentence.append(line)
            tmp_line = line.rstrip()
        else:
            tmp_line = line.rstrip()

    return dp_sentence, ori_sentence, correct_sentence

#alignedデータから3種類のタグ抽出
def parse_dp(dp_sentence):
    add_list = []
    msf_list = []
    oms_list = []

    fix_dp = dp_sentence.replace('<msf crr', '<msfcrr')
    re_add = re.findall('<add>[a-z]+</add>', fix_dp)
    re_msf = re.findall('<msfcrr[a-z"=>]+</msf>', fix_dp)
    re_oms = re.findall('<oms>[a-z]+</oms>', fix_dp)

    return re_add, re_msf, re_oms, fix_dp

pos_list = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS",\
        "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP",\
        "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB",\
        "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB"]

posa_dic = {}
for num, pos in enumerate(pos_list):
    posa_dic[pos] = num+1
#置換，脱落，余剰の操作抽出（内容語なら品詞，機能語なら単語）
#まずアライメントの情報を持ってきてからここに入れている
#POSはtreetaggerのposリストから
def detect_operate_pos(ori_sen, gec_sen, add, msf, oms, pos_dic, dp_sen):
    '''
    tagger = treetaggerwrapper.TreeTagger(TAGLANG='en',TAGDIR='/home/lr/hayashi/ra_web_app')
    # [単語\t品詞\t原形, .... のような形式]
    # item.split('\t')[0] -> 単語
    # item.split('\t')[1] -> 品詞
    ori_tagged = tagger.TagText(ori_sen)
    gec_tagged = tagger.TagText(gec_sen)

    ori_sen_list = ori_sen.split()
    gec_sen_list = gec_sen.split()

    ori_pos_list = [x.split('\t')[1] for x in ori_tagged]
    gec_pos_list = [x.split('\t')[1] for x in gec_tagged]
    '''
    #nltkのタガーで
    ori_tagged = pos_tag(wt(ori_sen))
    gec_tagged = pos_tag(wt(gec_sen))
    #print(www)
    ori_sen_list = ori_sen.split()
    gec_sen_list = gec_sen.split()
    ori_pos_list = [x[1] for x in ori_tagged]
    gec_pos_list = [x[1] for x in gec_tagged]

    add_pos = []
    msf_pos = []
    oms_pos = []
    add_words = []
    msf_words = []
    oms_words = []
    operation_word = []
    #更新が必要

    function_list = ["CC", "DT", "EX", "IN", "MD", "PDT", "POS", "PRP", \
                   "PRP$", "RP", "TO", "WDT", "WP", "WP$", "WRB"]


    #add/msf/oms_word = タグ付き<add>xxx</add>
    #中身と単語を特定（機能語なら単語，内容語なら品詞）したい
    #add:gec後の文章から抽出する


    ori_w_tag = []
    gec_w_tag = []
    for word in dp_sen.split():
        if '<add>' in word:
            gec_w_tag.append(word)
        elif '<oms>' in word:
            ori_w_tag.append(word)
        elif '<msrcrr' in word:
            ori_w_tag.append(word)
            gec_w_tag.append(word)
        else:
            ori_w_tag.append(word)
            gec_w_tag.append(word)

    for ori_word, ori_tag_word, ori_pos in zip(ori_sen_list, ori_w_tag, ori_pos_list):
        print(ori_word, ori_tag_word, ori_pos)
       # for add_word in add:
       #     print(add_word)
            #単語のマッチ
            #for pos, word in zip(gec_pos_list, gec_sen_list):
            #    print(pos, word, add_word)




    '''
            for gec_pos in gec_pos_list:
                if add_word.string == gec_pos[0]:
                    if gec_pos[1] == ".":
                        pass
                    elif gec_pos[1] not in pos_dic.keys():
                        add_pos.append("OTHER")
                    elif gec_pos[1] in function_list:
                        add_words.append(str(gec_pos[0].lower()))
                    else:
                        add_pos.append(gec_pos[1])
                        operation_word.append(str(gec_pos[0]))
        print(add_pos)
        print(add_words)
        print(operation_word)

    '''
    '''
    if len(msf) != 0:
        for msf_word in msf:
            #cor_pos_list
            for cor_pos in cor_pos_list:
                if msf_word.string == cor_pos[0]:
                    if cor_pos[1] == ".":
                        pass
                    elif cor_pos[1] not in pos_dic.keys():
                        msf_pos.append("OTHER")
                    elif cor_pos[1] in function_list:
                        msf_words.append(str(cor_pos[0].lower()))
                    else:
                        msf_pos.append(cor_pos[1])
                        operation_word.append(str(cor_pos[0]))

    if len(oms) != 0:
        for oms_word in oms:
            #ori_pos_list
            for ori_pos in ori_pos_list:
                if oms_word.string == ori_pos[0]:
                    if ori_pos[1] == ".":
                        pass
                    elif ori_pos[1] not in pos_dic.keys():
                        oms_pos.append("OTHER")
                    elif ori_pos[1] in function_list:
                        oms_words.append(str(ori_pos[0].lower()))
                    else:
                        oms_pos.append(ori_pos[1])
                        operation_word.append(str(ori_pos[0]))
    return add_pos, msf_pos, oms_pos, operation_word, ori_pos_list, \
            add_words, msf_words, oms_words
    '''

def main(args):

    if args.MODE == 'train':
        #ファイルもってくる&シャッフルする
        import glob
        files = glob.glob('../cefrj/original/**/*.raw')
        shuf_list = random.sample(files, len(files))
        x = []
        y = []
        for dat in shuf_list:
            print(dat)
            data = ''
            with open(dat,'r') as f:
                for i in f:
                    data += i.rstrip() + ' '

            #surface = Surface(unicode(data))
            surface = Surface(str(data))
            ngram, stats, diff = surface.features()
            #grmitem = GrmItem(unicode(data))
            grmitem = GrmItem(str(data))
            grm, pos_ngram, use_list = grmitem.features()
            inputs = Feature(ngram=ngram, pos_ngram=pos_ngram, grmitem=grm, word_difficulty=diff, stats=stats).concat()
            x.append(inputs)
            if 'A1' in dat:
                y.append(1)
            elif 'A2' in dat:
                y.append(2)
            elif 'B1' in dat:
                y.append(3)

        input_x = np.array(x)
        input_y = np.array(y)
        print(input_x.shape)
        print(input_y.shape)

        #学習
        clf = mord.LogisticAT(alpha=0.01)
        clf.fit(input_x, input_y)

        #モデル書き出し
        joblib.dump(clf, open(args.OUT, 'wb'))

    elif args.MODE == 'train_gec':
        #ファイルもってくる&シャッフルする
        #xmlファイルからもってくる
        import glob
        files = glob.glob('../cefrj/original_gec_pairs_xml/**/*.out')
        shuf_list = random.sample(files, len(files))
        x = []
        y = []

        #xmlデータ読み込み（入力，出力，アライメント結果）
        for dat in shuf_list:
            print(dat)
            with open(dat,'r') as f_xml:
                 aligned, original, gec_out = extract_dp_sentence(f_xml)

            '''
            #入力文に対しての処理
            original_text = ''
            for text in original:
                original_text += text.rstrip() + ' '

            surface = Surface(str(original_text))
            ngram, stats, diff = surface.features()
            grmitem = GrmItem(str(original_text))
            grm, pos_ngram, use_list = grmitem.features()

            #出力文に対しての処理
            gec_text = ''
            for text in gec_out:
                gec_text += text.rstrip()  + ' '
            grmitem_gec = GrmItem(str(gec_text))
            grm_gec, pos_ngram2_gec,  use_list_gec = grmitem_gec.features()
            print(grm_gec)
            #print(len(use_list))
            #print(len(use_list2))
            '''
            #置換，脱落，余剰検出
            aligned_text = ''
            #for text in aligned:
            #    aligned_text += text.rstrip() + ' '
            for o, g, a in zip(original, gec_out, aligned):
                add, msf, oms, dp_sen = parse_dp(a)
                detect_operate_pos(o, g, add, msf, oms, posa_dic, dp_sen)


        '''
            inputs = Feature(ngram=ngram, pos_ngram=pos_ngram, grmitem=grm, word_difficulty=diff, stats=stats).concat()
            x.append(inputs)
            if 'A1' in dat:
                y.append(1)
            elif 'A2' in dat:
                y.append(2)
            elif 'B1' in dat:
                y.append(3)

        input_x = np.array(x)
        input_y = np.array(y)
        print(input_x.shape)
        print(input_y.shape)

        #学習
        clf = mord.LogisticAT(alpha=0.01)
        clf.fit(input_x, input_y)

        #モデル書き出し
        joblib.dump(clf, open(args.OUT, 'wb'))
        '''

    elif args.MODE == 'test':
        #データ読み込み
        data = ''
        with open(args.INPUT,'r') as f:
            for i in f:
                data += i.rstrip() + ' '

        #素性作成
        #surface = Surface(unicode(data))
        surface = Surface(str(data))
        ngram, stats, diff = surface.features()
        #grmitem = GrmItem(unicode(data))
        grmitem = GrmItem(str(data))
        grm, pos_ngram, grm_freq = grmitem.features()
        inputs = Feature(ngram=ngram, pos_ngram=pos_ngram, grmitem=grm, word_difficulty=diff, stats=stats).concat()

        #モデル読み込み
        clf = joblib.load("./test.pkl")
        grade = clf.predict(inputs)
        print(output(grade, stats, diff, grm_freq))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--MODE', required=True, choices=['train', 'train_gec', 'test'])
    parser.add_argument('-o', '--OUT')
    parser.add_argument('-i', '--INPUT')
    args = parser.parse_args()
    main(args)
