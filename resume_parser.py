from turtle import home
import spacy
import io
import os
import util
import pprint
import multiprocessing as mp
import jieba
import sys
import pickle
import load_conf
import utils.data
import torch
import torch.autograd as autograd
import torch.optim as optim
from tqdm import tqdm

conf_path = './wclstm_ner.conf'
conf_dict = load_conf.load_conf(conf_path)
test_file = conf_dict['test']
model_dir = conf_dict['load_model']
dset_dir = conf_dict['save_dset']
output_file = conf_dict['output']
seg = conf_dict['seg']
status = conf_dict['status']
model_name = conf_dict['model']
save_model_dir = conf_dict['save_model']
gpu = torch.cuda.is_available()

char_emb = conf_dict['char_emb']
bichar_emb = conf_dict['bichar_emb']
gaz_file = conf_dict['gaz_file']

class ResumeParser(object):
    def __init__(self, resume):
        nlp = spacy.blank('en')
        self.__details = {
            'name_zh'               : None,
            'dob_zh'                : None,
            'gender_zh'             : None,
            'political_face'        : None,
            'highest_degree'        : None,
            'all_college'           : None,
            'all_profes'            : None,
            'all_edu'               : None,
            'undergraduate_school'  : None,
            'undergraduate_major'   : None,
            'master_school'         : None,
            'master_major'          : None,
            'doctoral_school'       : None,
            'doctoral_major'        : None,
            'is_beijing'            : None,
            'hometown'              : None,
            'phone_number'          : None,
            'mail'                  : None,  
        }

        self.__resume = resume

        #   提取文件后缀
        if not isinstance(self.__resume, io.BytesIO):
            ext = os.path.splitext(self.__resume)[1].split('.')[1]
        else:
            ext = self.__resume.name.split('.')[1]

        self.__pdf_text = util.pdf_text(self.__resume)
        #   line_text return a PATH
        self.__line_text_file = util.line_text(self.__resume)
        self.__clean_pdf_text = util.clear_text(self.__pdf_text)
        self.__jieba_text = jieba.cut(self.__clean_pdf_text, cut_all=True)
        self.__nlp = nlp(self.__clean_pdf_text)
        
        self.__data = util.load_data_setting(dset_dir)
        self.__data.generate_instance_with_gaz_3(self.__line_text_file, 'test')
        self.__pred_results = util.load_model_decode(model_dir, self.__data, 'test', gpu, seg)
        #   labeled txt
        self.__labeled_data_file = self.after_pred_txt(self.__pred_results, self.__line_text_file)
        self.__get_basic_details()
    def after_pred_txt(self, pred_results, line_file):
        #   transform n-dimension pred_results to 1-dimension list
        pred_final = []
        for a in pred_results:
            for b in a:
                pred_final.append(b)
        remove_blank_lines_file = './remove_blank_lines_file.txt'
        add_label_file = './add_label_file.txt'
        file1 = open(line_file, 'r', encoding='utf-8')
        file2 = open(remove_blank_lines_file, 'w', encoding='utf-8')
        file3 = open(add_label_file, 'w', encoding='utf-8')

        for line in file1.readlines():
            if line == '\n':
                line = line.strip('\n')
            file2.write(line)
        file1.close()
        file2.close()
        in_lines = open(remove_blank_lines_file, 'r', encoding='utf-8').readlines()
        for i in range(min(len(in_lines), len(pred_final))):
            label = pred_final[i]
            line = in_lines[i].replace('\n', ' ' + label + '\n')
            file3.write(line)
        file3.close()
        return add_label_file
    def get_extracted_data(self):
        return self.__details
    def get_pdftext(self):
        return self.__pdf_text
    def get_clean_pdftext(self):
        return self.__clean_pdf_text
    def __get_basic_details(self):
        name_zh = util.extra_name_zh(self.__pdf_text, self.__resume)
        all_college, all_edu, all_profes = util.extra_all_information(self.__labeled_data_file)
        dob_zh = util.get_dob_zh(self.__nlp, self.__clean_pdf_text)
        gender_zh = util.get_gender_zh(self.__nlp, name_zh)
        political_face = util.get_political_face(self.__nlp)
        highest_degree = util.get_highest_degree(self.__nlp)
        undergraduate_school = util.get_undergraduate_school(all_college, all_edu)
        undergraduate_major = util.get_undergraduate_major(all_profes, all_edu, all_college)
        master_school = util.get_master_school(all_college, all_edu)
        master_major = util.get_master_major(all_profes, all_edu, all_college)
        doctoral_school = util.get_doctoral_school(all_college, all_edu)
        doctoral_major = util.get_doctoral_major(all_profes, all_edu, all_college)
        hometown = util.get_hometown(self.__nlp)
        is_beijing = util.get_is_beijing(hometown)
        phone_number = util.get_phone_number(self.__clean_pdf_text)
        mail = util.get_mail(self.__clean_pdf_text)

        if highest_degree == '本科':
            master_school, master_major, doctoral_major, doctoral_school = '未知', '未知', '未知', '未知'
        if highest_degree == '硕士':
            doctoral_major, doctoral_school = '未知', '未知'

        self.__details['name_zh'] = name_zh
        self.__details['dob_zh'] = dob_zh
        self.__details['gender_zh'] = gender_zh
        self.__details['political_face'] = political_face
        self.__details['highest_degree'] = highest_degree
        self.__details['all_edu'] = all_edu
        self.__details['all_college'] = all_college
        self.__details['all_profes'] = all_profes
        self.__details['undergraduate_school'] = undergraduate_school
        self.__details['undergraduate_major'] = undergraduate_major
        self.__details['master_school'] = master_school
        self.__details['master_major'] = master_major
        self.__details['doctoral_school'] = doctoral_school
        self.__details['doctoral_major'] = doctoral_major
        self.__details['is_beijing'] = is_beijing
        self.__details['hometown'] = hometown
        self.__details['phone_number'] = phone_number
        self.__details['mail'] = mail

        return 

def resume_result_wrapper(resume):
        parser = ResumeParser(resume)
        return parser.get_extracted_data()

# if __name__ == '__main__':
#     pool = mp.Pool(mp.cpu_count())

#     resumes = []
#     data = []
#     for root, directories, filenames in os.walk('resumes'):
#         for filename in filenames:
#             file = os.path.join(root, filename)
#             resumes.append(file)

#     results = [pool.apply_async(resume_result_wrapper, args=(x,)) for x in resumes]

#     results = [p.get() for p in results]

#     pprint.pprint(results)
#     print('text-------------')