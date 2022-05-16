import pdftotext
from cocoNLP.extractor import extractor
import re
import spacy
import fitz
import ngender
import pandas as pd
from fuzzywuzzy import fuzz
from ltp import LTP
import pickle
import load_conf
import sys
import torch
import torch.autograd as autograd
import torch.optim as optim
from tqdm import tqdm
import random
import numpy as np
import time
from model.bilstmcrf import BiLSTM_CRF
from model.CNNmodel import CNNmodel
from model.cw_ner.lw.cw_ner import CW_NER

from utils.data import Data
from utils.metric import get_ner_fmeasure
seed_num = 100
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

ltp = LTP()

nlp = spacy.blank('en')
all_political_face = ['中共党员', '党员', '中共预备党员', '预备党员', '共青团员', '团员', '民革党员', '民盟盟员', '民建会员', '民进会员', '农工党党员', '致公党党员', '九三学社社员', '台盟盟员', '无党派人士', '群众']
#   1999-01-01 1999.01.01#
#   2000-01-01
re_str = {
    'dob': r'[1,2]\d{3}[-,.]\d?\d[-,.]?\d?\d?',
    'phone_number': r'1[35789]\d{1}-?\s*\d{4}-?\s*\d{4}',
    'mail': r'([^@|\s]+@[^@]+\.[^@|\s]+)',
    'school':r'[\u4e00-\u9fa5]*[\u5927\u5b66]'
}
def pdf_text(pdf_path):
    with open(pdf_path, 'rb') as f:
        pdf = pdftotext.PDF(f)
        return '\n\n'.join(pdf)
def get_text(file_path):
    with fitz.open(file_path) as f:
        text = ''
        for page in f:
            text += page.getText().strip()
        return text
def label_txt(file_path):
    return 

def line_text(pdf_path):
    results = []
    output_path = './pre/' + pdf_path.split('/')[1] + '_onepiece.txt'
    fp = open(output_path, 'w', encoding='utf-8')
    str = get_text(pdf_path)
    str = str.replace(' ', '')
    str = '\tO\n'.join(str)
    results.append(str)
    fp.write(str)
    fp.write('\tO\n')
    fp.close()
    return output_path

def clear_text(text):
    result = text
    for i in ['\n', ';', ',', '；', '，', '。', ':',  '|', '-', '|', '(', ')', '：', '（', '）', '丨']:
        result = result.replace(i, ' ')

    return result

def extra_name_zh(text, pdf_name):
    ex = extractor()
    return ex.extract_name(pdf_name)

def extra_name(labeled_data_path):
    name_inlines = open(labeled_data_path, 'r', encoding='utf-8').readlines()
    name_list = []
    for i in range(len(name_inlines)):
        ch = ''
        end = 0
        if name_inlines[i].split(' ')[-1].startswith('B-NAME'):
            ch = ch + name_inlines[i].split(' ')[0]
            for j in range(1,4):
                if name_inlines[i+j].split(' ')[-1].startswith('E-NAME'):
                    for k in range(1,j+1):
                        ch = ch + name_inlines[i + k].split(' ')[0]
                    for char in ['\t', 'O']:
                        ch = ch.replace(char, '')
                    name_list.append(ch)
    return name_list[0]

def get_dob_zh(nlp_text, text):
    dob_zh = '未知'
    dob_zh_token = show_tokens(nlp_text)
    dob_zh = re.findall(re_str['dob'], text)
    if dob_zh:
        dob_zh = ''.join(dob_zh[0])
    else:
        for i in range(len(dob_zh_token)):
            similarity = [fuzz.ratio(dob_zh_token[i], '出生年月日'), fuzz.ratio(dob_zh_token[i], '出生年月'), fuzz.ratio(dob_zh_token[i], '出生日期'), fuzz.ratio(dob_zh_token[i], '生日')]
            if any([s >= 90 for s in similarity]):
                dob_zh = dob_zh_token[i+1] if(dob_zh_token[i+1] != ' ') else dob_zh_token[i+2]
    return dob_zh
def get_gender_zh(nlp_text, name):
    if not name:
        return '未知'
    gender = '未知'
    gender_token = show_tokens(nlp_text)
    for i in range(len(gender_token)):
        if gender_token[i] in ['男', '男性', '男士']:
            gender = '男'
            break
        if gender_token[i] in ['女', '女性', '女士']:
            gender = '女'
            break

    if gender == '未知':
        if ngender.guess(name)[0] == 'female' and ngender.guess(name)[-1] > 0.5:
            gender = '女'
        else:
            gender = '男'

    return gender

def get_political_face(nlp_text):
    political_face = '未知'
    political_face_token = show_tokens(nlp_text)
    for i in range(len(political_face_token)):
        similarity = [fuzz.ratio(political_face_token[i], '政治面貌'), fuzz.ratio(political_face_token[i], '政治身份')]
        #   检索出“政治面貌”关键字，i+1或i+2为“党员”
        if any([s >= 90 for s in similarity]):
            political_face = political_face_token[i+1] if (political_face_token[i+1] in all_political_face) else political_face_token[i+2]
            break
        #   检索出“党员”
        if political_face_token[i] in all_political_face:
            political_face = political_face_token[i]
            break

    return political_face

def get_highest_degree(nlp_text):
    degree = '未知'
    degree_list = [1]
    find_degree_1 = ['本科', '学士']
    find_degree_2 = ['硕士', '研究生']
    result = {1: '本科', 2:'硕士' , 3:'博士' , 4:'博士后'}
    degree_token = show_tokens(nlp_text)
    for i in range(len(degree_token)):
        if '博士后' in degree_token[i]:
            degree_list.append(4)
        if '博士' in degree_token[i]:
            degree_list.append(3)
        if any(degree in degree_token[i] for degree in find_degree_2):
            degree_list.append(2)
        if any(degree in degree_token[i] for degree in find_degree_1):
            degree_list.append(1)
        degree = result[max(degree_list)]
    return degree

def get_undergraduate_school(college_list, edu_list):
    edu_under_list = ['本科','学士']
    edu_pos = 0
    for key in edu_list:
        if any([des in key for des in edu_under_list]):
            edu_pos = edu_list[key]
            break
    if college_list:
        for key in college_list:
            college_list[key] = abs(college_list[key] - edu_pos)

        return min(college_list, key=college_list.get)
    else:
        return '未知'

def get_undergraduate_major(profes_list, edu_list, college_list):
    edu_under_list = ['本科', '学士']
    edu_pos = 0
    for key in edu_list:
        if any([des in key for des in edu_under_list]):
            edu_pos = edu_list[key]
            break
    if profes_list:
        for key in profes_list:
            profes_list[key] = abs(profes_list[key] - edu_pos)
        return min(profes_list, key = profes_list.get)
    else:
        if college_list:
            for key in college_list:
                college_list[key] = abs(college_list[key] - edu_pos)
            return min(college_list, key = college_list.get)
        else:
            return '未知'


def get_master_school(college_list, edu_list):
    edu_master_list = ['硕士', '研究生']
    edu_pos = 0
    for key in edu_list:
        if any(des in key for des in edu_master_list):
            edu_pos = edu_list[key]
            break
    if college_list:
        for key in college_list:
            college_list[key] = abs(college_list[key] - edu_pos)

        return min(college_list, key = college_list.get)
    else:
        return '未知'

def get_master_major(profes_list, edu_list, college_list):
    edu_master_list = ['硕士', '研究生']
    edu_pos = 0
    for key in edu_list:
        if any(des in key for des in edu_master_list):
            edu_pos = edu_list[key]
            break
    if profes_list:
        for key in profes_list:
            profes_list[key] = abs(profes_list[key] - edu_pos)
        return min(profes_list, key = profes_list.get)
    else:
        if college_list:
            for key in college_list:
                college_list[key] = abs(college_list[key] - edu_pos)
            return min(college_list, key = college_list.get)
        else:
            return '未知'

def get_doctoral_school(college_list, edu_list):
    edu_phd_list = ['博士']
    edu_pos = 0
    for key in edu_list:
        if any(des in key for des in edu_phd_list):
            edu_pos = edu_list[key]
            break
    if college_list:
        for key in college_list:
            college_list[key] = abs(college_list[key] - edu_pos)
        return min(college_list, key = college_list.get)
    else:
        return '未知'

def get_doctoral_major(profes_list, edu_list, college_list):
    edu_phd_list = ['博士']
    edu_pos = 0
    for key in edu_list:
        if any(des in key for des in edu_phd_list):
            edu_pos = edu_list[key]
            break
    if profes_list:
        for key in profes_list:
            profes_list[key] = abs(profes_list[key] - edu_pos)
        return min(profes_list, key = profes_list.get)
    else:
        if college_list:
            for key in college_list:
                college_list[key] = abs(college_list[key] - edu_pos)
            return min(college_list, key = college_list.get)
        else:
            return '未知'


def get_is_beijing(hometown):
    if not hometown or hometown == '未知':
        return '未知'
    else:
        return '是京籍' if '北京' in hometown else '非京籍'
        
def get_hometown(nlp_text):
    hometown = '未知'
    hometown_token_list = show_tokens(nlp_text)
    for i in range(len(hometown_token_list)):
        similarity = [fuzz.ratio(hometown_token_list[i], '籍贯'), fuzz.ratio(hometown_token_list[i], '家乡'), fuzz.ratio(hometown_token_list[i], '现居城市')]
        if any([s >= 90 for s in similarity]):
            hometown = hometown_token_list[i+1] if (hometown_token_list[i+1] not in [' ', ':', '：']) else hometown_token_list[i+2]
            break

    return hometown

def get_phone_number(text):
    phone = re.findall(re_str['phone_number'], text)
    if phone:
        number = ''.join(phone[0])
        if len(number) > 10:
            return number
        else:
            return ''

def get_mail(text):
    email = re.findall(re_str['mail'], text)
    if email:
        try:
            return email[0].split()[0].strip(';')
        except IndexError:
            return 'None'
    else:
        return '未知'

def show_tokens(text):
    cv_tokens_list=[]
    cv_tokens_list.clear()
    for token in text:
        cv_tokens_list.append(token.text)
    
    return cv_tokens_list

def extra_all_college(text):
    college = list(set(re.findall(re_str['school'], text)))
    global college_list
    college_list = college
    return college
def extra_all_college_1(labeled_data_path):
    college_inlines = open(labeled_data_path, 'r', encoding='utf-8').readlines()
    college_list_with_pos = {}
    for i in range(len(college_inlines)):
        ch = ''
        start, end, avg_pos = 0, 0, 0
        if college_inlines[i].split(' ')[-1].startswith('B-ORG'):
            start = i
            ch = ch + college_inlines[i].split(' ')[0]
            for j in range(1,15):
                if college_inlines[i+j].split(' ')[-1].startswith('E-ORG'):
                    end = i + j
                    avg_pos = start + ((end - start) >> 1)
                    for k in range(1, j+1):
                        ch = ch + college_inlines[i+k].split(' ')[0]
                    for char in ['\t', 'O']:
                        ch = ch.replace(char, '')
                    college_list_with_pos[ch] = avg_pos

    return college_list_with_pos

#   提取 NAME, EDU, ORG, PRO
def extra_all_information(labeled_data_path):
    inlines = open(labeled_data_path, 'r', encoding='utf-8').readlines()
    name_list, college_list, edu_list, profes_list = [], {}, {}, {}
    for i in range(len(inlines)-15):
        ch = ''
        start, end, avg_pos = 0, 0, 0
        if inlines[i].split(' ')[-1].startswith('B-NAME'):
            ch = ch + inlines[i].split(' ')[0]
            for j in range(1,4):
                if inlines[i+j].split(' ')[-1].startswith('E-NAME'):
                    for k in range(1, j+1):
                        ch = ch + inlines[i + k].split(' ')[0]
                    for char in ['\t', 'O']:
                        ch = ch.replace(char, '')
                    name_list.append(ch)

        if inlines[i].split(' ')[-1].startswith('B-ORG'):
            start = i
            ch = ch + inlines[i].split(' ')[0]
            for j in range(1,15):
                if inlines[i+j].split(' ')[-1].startswith('E-ORG') :
                    end = i + j
                    avg_pos = start + ((end - start) >> 1)
                    for k in range(1,j+1):
                        ch = ch + inlines[i + k].split(' ')[0]
                    for char in ['\t', 'O']:
                        ch = ch.replace(char, '')
                    college_list[ch] = avg_pos

        if inlines[i].split(' ')[-1].startswith('B-EDU'):
            start = i
            ch = ch + inlines[i].split(' ')[0]
            for j in range(1,5):
                if inlines[i+j].split(' ')[-1].startswith('E-EDU'):
                    end = i + j 
                    avg_pos = start + ((end - start) >> 1)
                    for k in range(1, j+1):
                        ch = ch + inlines[i+k].split(' ')[0]
                    for char in ['\t', 'O']:
                        ch = ch.replace(char, '')
                    edu_list[ch] = avg_pos 
        if inlines[i].split(' ')[-1].startswith('B-PRO'):
            start = i
            ch = ch + inlines[i].split(' ')[0]
            for j in range(1,10):
                if inlines[i+j].split(' ')[-1].startswith('E-PRO'):
                    end = i + j
                    avg_pos = start + ((end - start) >> 1)
                    for k in range(1, j+1):
                        ch = ch + inlines[i+k].split(' ')[0]
                    for char in ['\t', 'O']:
                        ch = ch.replace(char, '')
                    profes_list[ch] = avg_pos

    edu_des_list = ['本科', '硕士', '研究生', '学士', '博士', '博士后']
    college_list_last = {}
    edu_list_last = {}
    for key in college_list:
            # if '大学' in key or '学院' in key:
        if key.endswith(('大学', '学院')):
            college_list_last[key] = college_list[key]  
    for key in edu_list:
        if any(des in key for des in edu_des_list):
            edu_list_last[key] = edu_list[key]
    for char in name_list:
        if (len(char) > 4 or len(char) < 2):
            name_list.remove(char)

    name_zh = name_list[0] if name_list else '未知'
    return college_list_last, edu_list, profes_list

def load_data_setting(save_file):
    with open(save_file, 'rb') as fp:
        data = pickle.load(fp)
    print("Data setting loaded from file: ", save_file)
    data.show_data_summary()
    return data
def batchify_with_label_3(input_batch_list, gpu, volatile_flag=False):
    batch_size = len(input_batch_list)
    chars = [sent[0] for sent in input_batch_list]
    bichars = [sent[1] for sent in input_batch_list]
    gazs = [sent[2] for sent in input_batch_list]
    reverse_gazs = [sent[3] for sent in input_batch_list]
    labels = [sent[4] for sent in input_batch_list]
    char_seq_lengths = torch.LongTensor(list(map(len, chars)))
    max_seq_len = char_seq_lengths.max()
    char_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
    bichar_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
    label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len))).byte()
    for idx, (seq, biseq, label, seqlen) in enumerate(zip(chars, bichars, labels, char_seq_lengths)):
        char_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        bichar_seq_tensor[idx, :seqlen] = torch.LongTensor(biseq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.Tensor([1] * int(seqlen))
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    bichar_seq_tensor = bichar_seq_tensor[char_perm_idx]
    label_seq_tensor = label_seq_tensor[char_perm_idx]
    mask = mask[char_perm_idx]

    _, char_seq_recover = char_perm_idx.sort(0, descending=False)

    gaz_list = [gazs[i] for i in char_perm_idx]
    reverse_gaz_list = [reverse_gazs[i] for i in char_perm_idx]

    if gpu:
        char_seq_tensor = char_seq_tensor.cuda()
        bichar_seq_tensor = bichar_seq_tensor.cuda()
        char_seq_lengths = char_seq_lengths.cuda()
        char_seq_recover = char_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        mask = mask.cuda()

    return gaz_list, reverse_gaz_list, char_seq_tensor, bichar_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask

def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """

    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = gold_variable.size(0)
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        # print "p:",pred, pred_tag.tolist()
        # print "g:", gold, gold_tag.tolist()
        assert (len(pred) == len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label

def evaluate(data, model, name):
    instances = []
    if name == 'test':
        instances = data.test_Ids
    pred_results = []
    gold_results = []

    # set model in eval model
    model.eval()
    batch_size = data.HP_batch_size
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num // batch_size + 1
    for batch_id in range(total_batch):
        start = batch_id * batch_size
        end = (batch_id + 1) * batch_size
        if end > train_num:
            end = train_num
        instance = instances[start:end]
        if not instance:
            continue
        pred_label, gold_label = -1, -1
        if data.model_name == 'WC-LSTM_model':
            gaz_list, reverse_gaz_list, batch_char, batch_bichar, batch_charlen, batch_charrecover, batch_label, mask = batchify_with_label_3(instance,data.HP_gpu,data.HP_num_layer)

            tag_seq = model(gaz_list, reverse_gaz_list, batch_char, batch_charlen, mask)
            pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_charrecover)
        pred_results += pred_label
        gold_results += gold_label

    decode_time = time.time() - start_time
    speed = len(instances) / decode_time
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    return speed, acc, p, r, f, pred_results

def load_model_decode(model_dir, data, name, gpu, seg):
    print(type(data))
    data.HP_gpu = gpu
    print('Load model from file: ', model_dir)

    model = None
    if data.model_name == 'WC-LSTM_model':
        model = CW_NER(data)
    elif data.model_name == 'CNN_model':
        model = CNNmodel(data)
    elif data.model_name == 'LSTM_model':
        model = BiLSTM_CRF(data)
    assert (model is not None)
    model.load_state_dict(torch.load(model_dir))

    start_time = time.time()
    speed, acc, p, r, f, pred_results = evaluate(data, model, name)
    end_time = time.time()
    time_cost = end_time - start_time

    if seg:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (name, time_cost, speed, acc, p, r, f))
    else:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f" % (name, time_cost, speed, acc))
    return pred_results



