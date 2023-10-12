import math
import cv2
import torch
import numpy as np
import  pickle
import os
from PIL import Image
import math
import  jsonlines
import itertools
import pickle
unified2id = {"none":0, "per":1,"org":2,"misc":3, "loc":4,"parent":5,"siblings":6,"couple":7,"neighbor":8,"peer":9,
             "charges":10,"alumi":11,"alternate_names":12,"place_of_residence":13,"place_of_birth":14,"member_of":15,
              "subsidiary":16,"locate_at":17,"contain":18,"present_in":19,"awarded":20, "race":21,"religion":22,
             "nationality":23,"part_of":24,"held_on":25}

Pseudo_label_id =  {"none":0,"person":1,"organization":2,"location":3,"miscellaneous":4}
from transformers import BertTokenizer,RobertaTokenizer
from transformers import CLIPProcessor


def get_evaluate_spans(tags, length, token_range):
    '''for BIO tag'''
    spans = []
    start = -1
    for i in range(length):
        l, r = token_range[i]
        if tags[l] == -1:
            continue
        elif tags[l] == 1:
            if start != -1:
                spans.append([start, i - 1])
            start = i
        elif tags[l] == 0:
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length - 1])
    return spans


class Instance(object):

    def box_to_patch_idx(self,img_size,box,patch_number):
        #return the patch_idx for given box and img_szie

        x_1,y_1,x_2,y_2 = box
        #都需要向下取整
        img_wide,img_high = img_size
        begin_img_x = math.floor(patch_number*x_1/img_wide)
        begin_img_y = math.floor(patch_number* y_1/img_high)

        end_img_x = math.floor(patch_number * x_2 / img_wide)
        end_img_y = math.floor(patch_number * y_2 / img_high)
        # contras_image_tags[begin_img_x:end_img_x + 1, begin_img_y:end_img_y + 1] = 1
        return begin_img_x,begin_img_y,end_img_x,end_img_y



    def get_contras_tags(self,sentence_item,token_range, patch_number, max_sequence_len,img_size):
        patch_number = int(patch_number)
        patch_number_all  = patch_number*patch_number
        contras_image_tags = torch.zeros(patch_number_all, patch_number_all).long()

        contras_imagetotext_tags = torch.zeros(patch_number_all, max_sequence_len).long()
        contras_text_tags = torch.zeros(max_sequence_len, max_sequence_len).long()
        pos_text_idx = torch.zeros(max_sequence_len).long()
        pos_image_idx =torch.zeros(patch_number_all).long()
        pos_imagetotext_idx = torch.zeros(patch_number_all).long()

        # contras_text_tags[:, :] = -1
        contras_text_tags[:self.length, :self.length] = 0
        contras_imagetotext_tags[:, self.length] = 0

        entity_region_dict = {}
        for list_three in sentence_item["entity_region"]:

            entity_name,labels,box = list_three
            if labels in entity_region_dict:
                entity_region_dict[labels].append([entity_name,box])
            else:
                entity_region_dict[labels] = [[entity_name,box]]
        # print(entity_region_dict)
        # exit()

        pre_label  = torch.zeros(max_sequence_len).long()
        for key,dict_item_list in entity_region_dict.items():
            key = key.strip()
            # if key =="miscellaneous":
            #     continue
            Pseudo_label = Pseudo_label_id[key]
            for dict_item in dict_item_list:
                name,box=dict_item[0],dict_item[1]
                # print(dict_item)
                begin_img_x, begin_img_y, end_img_x, end_img_y = self.box_to_patch_idx(img_size, box, patch_number)
                flatten_image = torch.zeros(patch_number, patch_number).long()
                flatten_image [begin_img_x:end_img_x+1,begin_img_y:end_img_y+1] = 1
                flatten_image = torch.flatten(flatten_image, start_dim=0).unsqueeze(1)

                flatten_image_trans = torch.transpose(flatten_image,0,1)
                image_met = torch.matmul(flatten_image,flatten_image_trans)

                contras_image_tags = torch.where(contras_image_tags>0,contras_image_tags,image_met)
                # for index,i in enumerate(contras_image_tags):
                #     print(index)
                #     print(i)
                # print("11111111111111111111111111111111111111111111111111111")
                name_list = name.split()
                # 这里是image_to_text
                # print(name_list)
                for item_token in name_list:

                    if (item_token in (sentence_item["token"])) == True:
                        token_begin, token_end = token_range[sentence_item["token"].index(item_token)]
                        pre_label[token_begin:token_end+1] = Pseudo_label
                        new = torch.repeat_interleave(flatten_image, token_end + 1 - token_begin, dim=-1)
                        old = contras_imagetotext_tags[:,token_begin:token_end+1]
                        item_imagetotext =  torch.where(old>0,old,new)
                        contras_imagetotext_tags[:,token_begin:token_end+1] = item_imagetotext

                name_list_cc = [(item,item) for item in name_list]
                name_list_cc.extend(list(itertools.combinations(name_list, 2)))
                for token_1_2 in name_list_cc:
                    token_1,token_2 = token_1_2
                    if (token_1 in (sentence_item["token"]))==True and (token_2 in (sentence_item["token"]))==True:
                        (idx_1_begin, idx_1_end)= token_range[sentence_item["token"].index(token_1)]
                        (idx_2_begin, idx_2_end) = token_range[sentence_item["token"].index(token_2)]
                        #所有的wordpice都参与了考虑
                        contras_text_tags[idx_1_begin:idx_1_end + 1, idx_2_begin:idx_2_end + 1] = 1
                        contras_text_tags[idx_2_begin:idx_2_end + 1, idx_1_begin:idx_1_end + 1] = 1
                        pos_text_idx[idx_1_begin:idx_1_end+1] = 1
                        pos_text_idx[idx_2_begin:idx_2_end + 1]=1

        a = torch.sum(contras_image_tags,dim=-1).numpy().tolist()
        for idx,item in enumerate(a):
            if int(idx)>0:
                pos_image_idx[idx]=1
                pos_imagetotext_idx[idx]=1

        return contras_image_tags,contras_imagetotext_tags,contras_text_tags,\
               pos_image_idx,pos_imagetotext_idx,pos_text_idx,pre_label


    def __init__(self, sentence_pack,image_path,text_tokenizer,processor, args,clip_args,patch_number):
        # print(sentence_pack)
        # print(sentence_pack)
        # exit()
        # print(sentence_pack)
        self.sentence = " ".join(sentence_pack["token"])
        self.tokens = self.sentence.split()
        self.token_range = []
        self.bert_tokens = text_tokenizer.encode(self.sentence)

        # print(self.bert_tokens)
        # print(text_tokenizer.convert_ids_to_tokens(self.bert_tokens))
        # print(self.bert_tokens)

        self.length = len(self.bert_tokens)
        image_name = image_path+"/"+sentence_pack["img_id"]

        img = Image.open(image_name)
        img_size = img.size

        max_sequence_len = max(args.max_sequence_len, self.length)
        # print(max_sequence_len)
        # prin

        self.bert_tokens_padding = torch.zeros(max_sequence_len).long()
        self.pospeech_padding = torch.zeros(max_sequence_len).long()
        self.sen_length = len(self.tokens)
        self.entity_tags = torch.zeros(max_sequence_len).long()
        self.tags = torch.zeros(max_sequence_len, max_sequence_len).long()
        self.mask = torch.zeros(max_sequence_len)

        self.image_input = processor(images=img, return_tensors="pt")['pixel_values'].squeeze()




        for i in range(self.length):
            self.bert_tokens_padding[i] = self.bert_tokens[i]
        # print(self.bert_tokens_padding)
        # exit()
        self.mask[:self.length] = 1

        token_start = 1
        # print(self.tokens)
        for i, w, in enumerate(self.tokens):

            token_end = token_start + len(text_tokenizer.encode(w, add_special_tokens=False))
            self.token_range.append([token_start, token_end-1])
            token_start = token_end
        # print(self.length)
        # print(self.token_range)
        # assert self.length == self.token_range[-1][-1]+2
        # exit()
        self.entity_tags[self.length:] = -1
        self.entity_tags[0] = -1
        self.entity_tags[self.length-1] = -1

        self.tags[:, :] = -1
        for i in range(1, self.length-1):
            for j in range(i, self.length-1):
                self.tags[i][j] = 0
        #The three tags for  constrstive subtasks
        contras_image_tags, contras_imagetotext_tags, contras_text_tags,pos_image_idx,pos_imagetotext_idx,pos_text_idx,pre_label\
            = self.get_contras_tags(sentence_pack,self.token_range,patch_number,max_sequence_len,img_size)
        self.contras_image_tags=contras_image_tags
        self.contras_imagetotext_tags=contras_imagetotext_tags
        self.contras_text_tags=contras_text_tags
        self.pos_image_idx = pos_image_idx
        self.pos_imagetotext_idx = pos_imagetotext_idx
        self.pos_text_idx = pos_text_idx
        self.pseudo_label = pre_label
        # print(self.pseudo_label)
        # exit()
        if ('label_list' in sentence_pack.keys())==False:
            0
            #for pre_train


        else:
            for index, triple in enumerate(sentence_pack['label_list']):
                triple = triple[0]
                # print(triple)
                begin_entity_infor = triple['beg_ent']
                end_entity_infor = triple['sec_ent']
                relation_tags = int(unified2id[triple['relation'].lower()])
                '''set tag for begin_entity'''
                begin_entity_tags = int(unified2id[begin_entity_infor["tags"].lower()])
                begin_entity_span = begin_entity_infor["pos"]
                end_entity_tags = int(unified2id[end_entity_infor ["tags"].lower()])
                end_entity_span = end_entity_infor["pos"]

                l,r =begin_entity_span[0],begin_entity_span[1]-1
                start = self.token_range[l][0]
                end = self.token_range[r][1]
                # print(self.token_range)
                # print(start)
                # print(end)
                for i in range(start, end+1):
                    for j in range(i, end+1):
                        self.tags[i][j] = begin_entity_tags
                for i in range(l, r+1):
                    set_tag = begin_entity_tags
                    al, ar = self.token_range[i]
                    self.entity_tags[al] = set_tag
                    self.entity_tags[al+1:ar+1] = -1
                    # '''mask positions of sub words'''
                    self.tags[al+1:ar+1, :] = -1
                    self.tags[:, al+1:ar+1] = -1
                    #不 mask的方式
                    # self.entity_tags[al + 1:ar + 1] = set_tag
                    # '''mask positions of sub words'''
                    # self.tags[al + 1:ar + 1, :] = set_tag
                    # self.tags[:, al + 1:ar + 1] = set_tag

                '''set tag for end_entity'''
                l, r = end_entity_span[0],end_entity_span[1]-1
                start = self.token_range[l][0]
                end = self.token_range[r][1]
                for i in range(start, end+1):
                    for j in range(i, end+1):
                        self.tags[i][j] = end_entity_tags
                for i in range(l, r+1):
                    set_tag = end_entity_tags
                    pl, pr = self.token_range[i]
                    self.entity_tags[pl] = set_tag
                    #mask sub words的方式
                    self.entity_tags[pl+1:pr+1] = -1
                    self.tags[pl+1:pr+1, :] = -1
                    self.tags[:, pl+1:pr+1] = -1

                    #不masK的方式
                    # self.entity_tags[pl + 1:pr + 1] = set_tag
                    # self.tags[pl + 1:pr + 1, :] =set_tag
                    # self.tags[:, pl + 1:pr + 1] =set_tag

                al, ar =begin_entity_span[0],begin_entity_span[1]-1
                pl, pr= end_entity_span[0],end_entity_span[1]-1
                for i in range(al, ar+1):
                    for j in range(pl, pr+1):
                        sal, sar = self.token_range[i]
                        spl, spr = self.token_range[j]
                        #mask
                        self.tags[sal:sar+1, spl:spr+1] = -1
                        #不要mask的
                        # self.tags[sal:sar + 1, spl:spr + 1] =relation_tags
                        if i > j:
                            self.tags[spl][sal] = relation_tags
                        else:
                            self.tags[sal][spl] = relation_tags
        # for index,i in enumerate(self.tags):
        #     print(index)
        #     print(i)
        # print(self.tags)
        # exit()




def load_data_instances(text_path,image_path,args,clip_args):
    # print(text_path)
    file_name = text_path+".pickle"
    # if os.path.exists(file_name):
    #     # file = open('file_name', 'rb')
    #     # instances = pickle.load(file)
    #     # print(instances)
    #     print("1111111111111111111111111111")
    # else:
    text_file = open(text_path,"r",encoding="utf-8")
    sentence_packs = jsonlines.Reader(text_file)

    instances = list()

    text_tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer_path)
    # text_tokenizer = RobertaTokenizer.from_pretrained(args.roberta_tokenizer_path)

    processor = CLIPProcessor.from_pretrained(args.image_pretrained_path)
    image_size = clip_args.vision_config.image_size
    patch_size = clip_args.vision_config.patch_size

    patch_number = image_size/patch_size
    for index,sentence_pack in enumerate(sentence_packs):
        if (index)%100==0:
            print(index)
            # return
        try:
            a = Instance(sentence_pack,image_path,text_tokenizer,processor, args,clip_args,patch_number)
            instances.append(a)
        except:
            print("error")


    # with open(file_name, 'wb') as f:
    #     pickle.dump(instances, f)
    print("pre_train_data number is {0}".format(len(instances)))
    return instances


class DataIterator(object):
    def __init__(self, instances, args):
        self.instances = instances
        self.args = args
        self.batch_count = math.ceil(len(instances)/args.batch_size)

    def get_batch(self, index):
        sentences = []
        sens_lens = []
        token_ranges = []
        bert_tokens = []
        lengths = []
        token_masks = []
        entity_tags = []
        tags = []
        image_feature =[]
        contras_image_tags = []
        contras_imagetotext_tags = []
        contras_text_tags = []
        pos_image_idx=[]
        pos_imagetotext_idx=[]
        pos_text_idx=[]
        pseudo_label_idx = []

        for i in range(index * self.args.batch_size,
                       min((index + 1) * self.args.batch_size, len(self.instances))):
            # print(self.instances[i])
            # exit()
            sentences.append(self.instances[i].sentence)
            sens_lens.append(self.instances[i].sen_length)
            token_ranges.append(self.instances[i].token_range)

            bert_tokens.append(self.instances[i].bert_tokens_padding)
            lengths.append(self.instances[i].length)
            token_masks.append(self.instances[i].mask)
            # print(self.instances[i].image_input)
            image_feature.append(self.instances[i].image_input)
            entity_tags.append(self.instances[i].entity_tags)
            tags.append(self.instances[i].tags)
            contras_image_tags.append(self.instances[i].contras_image_tags)
            contras_imagetotext_tags.append(self.instances[i].contras_imagetotext_tags)
            contras_text_tags.append(self.instances[i].contras_text_tags)
            pos_image_idx.append(self.instances[i].pos_image_idx)
            pos_imagetotext_idx.append(self.instances[i].pos_imagetotext_idx)
            pos_text_idx.append(self.instances[i].pos_text_idx)
            pseudo_label_idx.append(self.instances[i].pseudo_label)


        bert_tokens = torch.stack(bert_tokens).to(self.args.device)
        lengths = torch.tensor(lengths).to(self.args.device)
        token_masks = torch.stack(token_masks).to(self.args.device)


        image_feature=torch.stack(image_feature).to(self.args.device)

        entity_tags = torch.stack(entity_tags).to(self.args.device)
        tags = torch.stack(tags).to(self.args.device)

        contras_image_tags = torch.stack(contras_image_tags).to(self.args.device)
        contras_imagetotext_tags =torch.stack(contras_imagetotext_tags).to(self.args.device)
        contras_text_tags = torch.stack(contras_text_tags).to(self.args.device)
        pos_image_idx = torch.stack(pos_image_idx).to(self.args.device)
        pos_imagetotext_idx=torch.stack(pos_imagetotext_idx).to(self.args.device)
        pos_text_idx = torch.stack(pos_text_idx).to(self.args.device)
        pseudo_label_idx = torch.stack(pseudo_label_idx).to(self.args.device)
        # print(bert_tokens.shape)
        # print(111111111111111)
        # exit()
        # exit()
        return  bert_tokens, lengths, token_masks, sens_lens, token_ranges, image_feature,entity_tags,tags,\
                contras_image_tags,contras_imagetotext_tags,contras_text_tags,pos_image_idx,pos_imagetotext_idx,pos_text_idx,pseudo_label_idx
