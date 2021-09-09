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

import nltk
from nltk.tokenize import sent_tokenize
from nltk import word_tokenize as wt
from nltk import pos_tag
from collections import Counter
from collections import OrderedDict
from sklearn.externals import joblib


#単語難易度読み込み
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
#素性として使うのでdic
function_word_dic = {}
with open("../dat/treetagger_function.word","r") as f:
    for num, i in enumerate(f):
        function_word_dic[str(i.rstrip())] = str(num+1)

#機能語品詞読み込み（treetagger:21種類)
#機能語だったら弾くのでリスト
function_pos_list = []
with open('../dat/treetagger_function.list', 'r') as f:
    for function_pos_word in f:
        function_pos_list.append(function_pos_word.rstrip())

#内容語読み込み（treetagger:37種類）
#素性として振り分けるのでdic
content_pos_dic = {}
with open('../dat/treetagger_content.list', 'r') as f:
    for num, i in enumerate(f):
        content_pos_dic[str(i.rstrip())] = str(num+1)

#文法項目の読み込み
grmlist = []
num_grm_dic = {}
num_list_dic = {}
with open('../dat/grmitem.txt', 'r') as f:
    for num, i in enumerate(f, 1):
        grmlist.append(i.rstrip().split('\t')[1])
        num_grm_dic[num] = i.rstrip().split('\t')[1]
        num_list_dic[num] = i.rstrip().split('\t')[0]

#表層情報
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

##GrmItem継承させる
#gec前後で文に対し抽出後差分を見る
class GrmItem_gec(GrmItem):
    pass

#素性作成用
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

#置換，脱落，余剰の操作抽出（内容語なら品詞，機能語なら単語）
#まずアライメントの情報を持ってきてからここに入れている
#POSはtreetaggerのposリストから
tagger = treetaggerwrapper.TreeTagger(TAGLANG='en',TAGDIR='/home/lr/hayashi/ra_web_app')
def detect_operate_pos(ori_sen, gec_sen, dp_sen, content_dic, function_dic, function_pos):
    #'''
    # [単語\t品詞\t原形, .... のような形式]
    # item.split('\t')[0] -> 単語
    # item.split('\t')[1] -> 品詞
    ori_tagged = tagger.TagText(ori_sen)
    gec_tagged = tagger.TagText(gec_sen)

    ori_sen_list = ori_sen.split()
    gec_sen_list = gec_sen.split()

    ori_pos_list = [x.split('\t')[1] for x in ori_tagged]
    gec_pos_list = [x.split('\t')[1] for x in gec_tagged]

    add_pos = []
    msf_pos = []
    oms_pos = []
    add_words = []
    msf_words = []
    oms_words = []
    operation_word = []

    dp_sen = dp_sen.replace('<msf crr', '<msfcrr')

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

    feature_len = len(content_dic) + len(function_dic)

    #機能語単語リスト(207単語)×3 + 内容語リスト(37種類)×3
    #oms_func -> oms_content -> add -> msf
    #オリジナル（oms)
    #print(content_dic.keys())
    out_list = []
    for ori_word, ori_tag_word, ori_pos in zip(ori_sen_list, ori_w_tag, ori_pos_list):
        if '<oms>' in ori_tag_word:
            #機能語であれば単語
            if ori_pos in function_pos:
                if ori_word.lower() in function_dic.keys():
                    out_list.append(int(function_dic[ori_word.lower()]))
            #内容語
            elif ori_pos in content_dic.keys():
                out_list.append(int(content_dic[ori_pos]) + len(function_dic))
            else:
                pass

    #gec後（add, msf)
    #msf(置換）は修正後のものを採用
    for gec_word, gec_tag_word, gec_pos in zip(gec_sen_list, gec_w_tag, gec_pos_list):
        if '<add>' in gec_tag_word:
            #print(gec_word, gec_tag_word, gec_pos)
            #機能語であれば単語
            if gec_pos in function_pos:
                if gec_word.lower() in function_dic.keys():
                    out_list.append(int(function_dic[gec_word.lower()]) + feature_len)
            #内容語
            elif gec_pos in content_dic.keys():
                out_list.append(int(content_dic[gec_pos]) + len(function_dic) + feature_len)
            else:
                pass

        if '<msfcrr' in gec_tag_word:
            #print(gec_word, gec_tag_word, gec_pos)
            #機能語であれば単語
            if gec_pos in function_pos:
                if gec_word.lower() in function_dic.keys():
                    out_list.append(int(function_dic[gec_word.lower()]) + 2*feature_len)
            #内容語
            elif gec_pos in content_dic.keys():
                out_list.append(int(content_dic[gec_pos]) + len(function_dic) + 2*feature_len)
            else:
                pass

    return out_list

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

            #置換，脱落，余剰検出
            #aligned_text = ''
            #for text in aligned:
            #    aligned_text += text.rstrip() + ' '
            original_text = ''
            operation_features = []
            for ori_sen, gec_sen, dp_sen in zip(original, gec_out, aligned):
                original_text += ori_sen.rstrip() + ' '
                #add, msf, oms, dp_sen = parse_dp(a)
                #内容語品詞dic, 機能語単語dic, 機能語品詞リスト
                aa = detect_operate_pos(ori_sen, gec_sen, dp_sen, content_pos_dic, function_word_dic, function_pos_list)
                operation_features.extend(aa)
            surface = Surface(str(original_text))
            ngram, stats, diff = surface.features()

            #ここで比較する
            #文法項目の件
            #修正前，修正後どちらでも正規表現の抽出が必要
            #同時にプログラムを回す
            #if 修正前後どちらもある→正しい利用
            #elif 修正前なし，修正後あり→誤用していた（文法項目+add or msf)
            #elif 修正前あり，修正後なし→誤用していた（文法項目+oms)
            grmitem = GrmItem(str(original_text))
            grm, pos_ngram, use_list = grmitem.features()
            print(Counter(operation_features))


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
