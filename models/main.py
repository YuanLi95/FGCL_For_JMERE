#coding utf-8
import copy
import os
import random
import argparse
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.nn.functional as F
from tqdm import trange
from transformers import  CLIPConfig
from utils.data import load_data_instances, DataIterator

from utils.utils import Metric

import  numpy as np
import  codecs
import  time
from  models.FGCL import  FGCLNetwork
from utils.Focal_loss import FocalLoss
def setup_seed(seed):
    # Set the random seed for the torch, numpy, random, and cudnn
    # deterministic settings
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)  # cpu
    # torch.cuda.manual_seed(seed)  # gpu
    # torch.cuda.manual_seed_all(seed)



def train(args,clip_args):

    # load dataset

    # text_train_path = args.prefix  + 'train_entity_region.json'
    # text_val_path = args.prefix + '/val_entity_region.json'
    # text_test_path = args.prefix +  '/test_entity_region.json'

    text_train_path = args.prefix + 'train_entity_region_spacy.json'
    text_val_path = args.prefix + '/val_entity_region_spacy.json'
    text_test_path = args.prefix + '/test_entity_region_spacy.json'
    image_train_path = args.image_path+"/train/"
    image_val_path = args.image_path + "/val"
    image_test_path = args.image_path + "/test/"


    instances_train = load_data_instances(text_train_path,image_train_path, args, clip_args)
    instances_val = load_data_instances(text_val_path,image_val_path, args, clip_args )
    instances_test = load_data_instances(text_test_path,image_test_path, args, clip_args )
    # exit()
    # random.shuffle(instances_train)


    trainset = DataIterator(instances_train, args)
    valset = DataIterator(instances_val, args)
    testset = DataIterator(instances_test, args)
    f_out = codecs.open('log/'  + ' {0}_val.txt'.format(args.model), 'a', encoding="utf-8")

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    model = FGCLNetwork(args,clip_args).to(args.device)
   # pre_train_model = "./savemodel/spacy/FGCLNetworktripletpretrain1.pt"
    #model.load_state_dict(torch.load(pre_train_model),strict=False)
    # model = model.load(pre_train_model, map_location=args.device)
    # print(model)
    # self_loss = FocalLoss(args).to(device=args.device)

    bert_params = list(map(id, model.bert.parameters()))
    # print(bert_params)
    clip_params = list(map(id, model.clip_image.parameters()))
    other_all_params  = clip_params+bert_params
    print(other_all_params)

    my_self_params = filter(lambda p: id(p) not in other_all_params,
                            model.parameters())

    # print(my_self_params)

    optimizer = torch.optim.Adam([{"params":my_self_params,"lr":args.lr},
                                  {"params": model.bert.parameters(), "lr": 2e-5},
                                  {"params": model.clip_image.parameters(), "lr": 2e-5}
                                  ])
    # optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": 2e-5},
                                  # {"params": model.clip_image.parameters(), "lr": 2e-5}
                                  # ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 64, eta_min=0, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decline, gamma=0.5, last_epoch=-1)

    best_joint_f1 = 0
    best_joint_epoch = 0
    test_f1 = 0
    test_p = 0
    test_r = 0
    best_test_model = None
    dev_test_model = None
    for indx in range(args.epochs):
        model.train()
        print('Epoch:{}'.format(indx))
        train_all_loss = 0.0
        train_ot_loss=0.0
        all_setps = 0
        for j in trange(trainset.batch_count):
            all_setps+=1
            bert_tokens, lengths, token_masks, sens_lens, token_ranges, image_feature, entity_tags, tags, \
            contras_image_tags, contras_imagetotext_tags, contras_text_tags,\
            pos_image_idx, pos_imagetotext_idx, pos_text_idx,pseudo_label_idx = trainset.get_batch(j)
            #
            # print(model)
            # exit()
            # print(111111111111111111111)
            # exit()
            if args.model=="FGCLNetwork":
                preds, align_loss,pre_entity = model(bert_tokens, lengths, token_masks, sens_lens,image_feature, \
                                           contras_image_tags, contras_imagetotext_tags,contras_text_tags,
                                          pos_image_idx,pos_imagetotext_idx,pos_text_idx,pseudo_label_idx
                                           )
            # print(122222222222222222222222222222222222222222222222)
            batch_max_lengths = torch.max(lengths)
            # preds =

            if len(preds.shape)==3:
                preds = preds.unsqueeze(0)
                # pre_entity = pre_entity.unsqueeze(0)
            preds = preds[:, :batch_max_lengths, :batch_max_lengths,:]
            tags = tags[:,:batch_max_lengths,:batch_max_lengths]
            entity_tags = entity_tags[:,:batch_max_lengths]
            pre_entity = pre_entity[:,:batch_max_lengths]
            # exit()
            preds_flatten = preds.reshape([-1, preds.shape[-1]])
            pre_entity_flatten = pre_entity.reshape([-1, pre_entity.shape[-1]])
            # print(preds)
            # print(tags.shape)
            # exit()
            tags_flatten = tags.reshape([-1])
            entity_tags = entity_tags.reshape([-1])
            # print(tags)
            # print(tags_flatten[0])
            # label_loss = self_loss(preds_flatten, tags_flatten)
            label_loss = F.cross_entropy(preds_flatten, tags_flatten, ignore_index=-1)
            entity_loss = F.cross_entropy(pre_entity_flatten, entity_tags, ignore_index=-1)
            # print(label_loss)

            loss = label_loss+args.alpha_align_loss * align_loss+args.alpha_entity_loss*entity_loss
            train_all_loss+=loss.item()
            train_ot_loss+= args.alpha_align_loss * align_loss
            optimizer.zero_grad()
            loss.backward()
            # loss.backward(retain_graph=True)

            optimizer.step()
        scheduler.step()
        print('this epoch train loss :{0}  ot_loss:{1}'.format(train_all_loss/all_setps,train_ot_loss/all_setps))
        # print("------------------this is train result-------------------------------------")
        # # _, _, _, _ = eval(model, trainset, args)
        #
        # print("-------this is trian result-------------------------------------")
        # _, _, _, _, _ = eval(model, trainset, args)
        print("------------------this is dev result-------------------------------------")
        joint_precision, joint_recall, joint_f1,dev_loss,dev_entity_result = eval(model, valset, args)
        print("------------------this is test result-------------------------------------")
        test_joint_precision, test_joint_recall, test_joint_f1, _,test_entity_result = eval(model, testset, args)
        if joint_f1 > best_joint_f1:
            best_joint_f1 = joint_f1
            best_joint_epoch = indx
            dev_test_model = copy.deepcopy(model)

        if test_joint_f1 > test_f1:
            test_f1 = test_joint_f1
            test_p = test_joint_precision
            test_r = test_joint_recall
            print("best test")
            best_test_model = copy.deepcopy(model)
        print("11111111111111")

        print('this poch:\t dev {} loss: {:.5f}\n\n'.format(args.task, dev_loss))
    model_path = args.model_dir + args.model + args.task + "dev_for_test_f1"+str(best_joint_f1) + "dev" + '.pt'

    torch.save(dev_test_model, model_path)


    best_test_model_path = args.model_dir + args.model + args.task  + "best_test_f1" + str(
        test_f1) + '.pt'
    torch.save(best_test_model, best_test_model_path)

    arguments = " "
    for arg in vars(args):
        if arg== "dependency_embedding":
            continue
        elif arg == "position_embedding":
            continue
        else:
            arguments += '{0}: {1} '.format(arg, getattr(args, arg))


    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

    f_out.write('time:{0}\n'.format(time_str))
    test_model = torch.load(model_path).to(args.device)



    dev_for_test_precision, dev_for_test_recall, dev_for_test_f1, _,dev_for_test_entity = eval(test_model, testset, args)

    best_dev_precision, best_dev_recall, best_dev_f1, _, best_dev_entity = eval(test_model, valset,
                                                                                                args)


    best_test_model_path = args.model_dir + args.model + args.task  + "best_test_f1" + str(test_f1) + '.pt'
    best_test_model =torch.load(best_test_model_path).to(args.device)
    best_test_precision, best_test_recall, best_test_f1, _, best_test_entity = eval(best_test_model, testset,
                                                                                                args)
    f_out.write(arguments)
    f_out.write("\n")
    f_out.write('dev_max_test_acc: {0}, dev_max_test_recall:{1}, dev_max_f1: {2}\n'.format(dev_for_test_precision,
                                                                                           dev_for_test_recall,
                                                                                             dev_for_test_f1))

    f_out.write('dev_max_test_acc_entity: {0}, dev_max_test_recall_entity:{1}, dev_max_f1_entity: {2}\n'.format(dev_for_test_entity[0],
                                                                                           dev_for_test_entity[1],
                                                                                           dev_for_test_entity[2],))

    f_out.write('best_dev_acc: {0}, best_dev_recall:{1}, best_dev_f1: {2}\n'.format(best_dev_precision,
                                                                                           best_dev_recall,
                                                                                           best_dev_f1))

    f_out.write('best_dev_acc_entity: {0}, best_dev_recall_entity:{1}, best_dev_entity: {2}\n'.format(
        best_dev_entity[0],
        best_dev_entity[1],
        best_dev_entity[2], ))





    f_out.write('best_test_precision: {0}, best_test_recall:{1}, best_test_f1: {2}\n'.format(best_test_precision,
                                                                                           best_test_recall,
                                                                                           best_test_f1))

    f_out.write('best_test_precision_entity: {0}, best_test_recall_entity:{1}, best_test_f1_entity: {2}\n'.format(
        best_test_entity[0],
        best_test_entity[1],
        best_test_entity[2], ))
    f_out.write("\n")

    f_out.close()
    print('best_test_precision: {0}, best_test_recall:{1}, best_test_f1: {2}\n'.format(best_test_precision,
                                                                                           best_test_recall,
                                                                                           best_test_f1))

    print('best epoch: {}\tbest dev {} f1: {:.5f}\n\n'.format(best_joint_epoch, args.task, best_test_f1))
    print('max test precision:{} ----- recall:{}-------- f1:{}'.format(str(test_p), str(test_r), str(test_f1)))


def pre_train(args,clip_args):

    # load dataset

    pre_train_text_path = args.pretrain_data  + 'pretrained_entity_region_spacy.json'

    pre_train_image_path = args.pretrain_image
    model = FGCLNetwork(args, clip_args).to(args.device)
    print(model)

    instances_pretrain = load_data_instances(pre_train_text_path,pre_train_image_path, args, clip_args)

    # exit()
    # random.shuffle(instances_train)


    trainset = DataIterator(instances_pretrain, args)

    f_out = codecs.open('log/'  + ' {0}_val.txt'.format(args.model), 'a', encoding="utf-8")

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)




    bert_params = list(map(id, model.bert.parameters()))
    # print(bert_params)
    clip_params = list(map(id, model.clip_image.parameters()))
    other_all_params  = clip_params+bert_params

    my_self_params = filter(lambda p: id(p) not in other_all_params,
                            model.parameters())

    # print(my_self_params)

    optimizer = torch.optim.Adam([{"params":my_self_params,"lr":args.lr},
                                  {"params": model.bert.parameters(), "lr": 2e-5},
                                  {"params": model.clip_image.parameters(), "lr": 2e-5}
                                  ])
    # optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": 2e-5},
                                  # {"params": model.clip_image.parameters(), "lr": 2e-5}
                                  # ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30, eta_min=0, last_epoch=-1)

    print(args.epochs)
    for indx in range(args.epochs):
        model.train()
        print('Epoch:{}'.format(indx))
        train_all_loss = 0.0
        train_ot_loss=0.0
        all_setps = 0
        for j in trange(trainset.batch_count):
            all_setps+=1
            bert_tokens, lengths, token_masks, sens_lens, token_ranges, image_feature, entity_tags, tags, \
            contras_image_tags, contras_imagetotext_tags, contras_text_tags,\
            pos_image_idx, pos_imagetotext_idx, pos_text_idx,pseudo_label_idx = trainset.get_batch(j)


            if args.model=="FGCLNetwork":
                align_loss= model.Pretrain(bert_tokens, lengths, token_masks, sens_lens,image_feature, \
                                           contras_image_tags, contras_imagetotext_tags,contras_text_tags,
                                          pos_image_idx,pos_imagetotext_idx,pos_text_idx,pseudo_label_idx
                                           )

            loss = align_loss
            train_all_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            # loss.backward(retain_graph=True)

            optimizer.step()
        scheduler.step()

        model_path = args.model_dir + args.model + args.task + "pretrain"+str(indx) + '.pt'

        torch.save(model.state_dict(), model_path)
        print('this epoch train loss :{0}  ot_loss:{1}'.format(train_all_loss / all_setps, train_ot_loss / all_setps))




def eval(model, dataset, args,print_result=False):
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []
        all_lengths = []
        all_sens_lengths = []
        all_token_ranges = []
        dev_loss =0.0
        steps = 0
        for i in range(dataset.batch_count):
            steps+=1
            bert_tokens, lengths, token_masks, sens_lens, token_ranges, image_feature, entity_tags, tags, \
            contras_image_tags, contras_imagetotext_tags, contras_text_tags, \
            pos_image_idx, pos_imagetotext_idx, pos_text_idx,pseudo_label_idx= dataset.get_batch(i)

            if args.model == "FGCLNetwork":
                prediction, align_loss,pre_entity = model(bert_tokens, lengths, token_masks, sens_lens, image_feature, \
                                               contras_image_tags, contras_imagetotext_tags, contras_text_tags,
                                               pos_image_idx, pos_imagetotext_idx, pos_text_idx,pseudo_label_idx
                                               )


            # prediction ,ot_loss = model(bert_tokens,  token_masks, token_dependency_masks, \
            # token_syntactic_position, token_edge_data, token_frequency_graph,pospeech_tokens, image_rel_matrix, image_rel_mask, image_feature,
            #               )
            # print(prediction.size())
            prediction_argmax = torch.argmax(prediction, dim=-1)
            tags_flatten = tags[:, :prediction.shape[1], :prediction.shape[1]].reshape([-1])
            prediction_flatten = prediction.reshape([-1, prediction.shape[-1]])
            dev_loss = dev_loss + F.cross_entropy(prediction_flatten, tags_flatten, ignore_index=-1)
            prediction_padded = torch.zeros(prediction.shape[0], args.max_sequence_len, args.max_sequence_len)
            prediction_padded[:, :prediction_argmax .shape[1], :prediction_argmax .shape[1]] =prediction_argmax

            all_preds.append(prediction_padded)
            all_labels.append(tags)
            all_lengths.append(lengths)
            all_sens_lengths.extend(sens_lens)
            all_token_ranges.extend(token_ranges)


        all_preds = torch.cat(all_preds, dim=0).cpu().tolist()
        all_labels = torch.cat(all_labels, dim=0).cpu().tolist()
        all_lengths = torch.cat(all_lengths, dim=0).cpu().tolist()

        metric = Metric(args, all_preds, all_labels, all_lengths, all_sens_lengths, all_token_ranges, ignore_index=-1)
        precision, recall, f1 = metric.score_uniontags(print_result)
        entity_results = metric.score_entity()
        print('entity_results\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(entity_results[0], entity_results[1],
                                                                  entity_results[2]))
        print("unified_results"+ '\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}\n'.format(precision, recall, f1))

    model.train()
    return precision, recall, f1, dev_loss,entity_results


def test(args,clip_args):
    print("Evaluation on testset:")
    text_test_path = args.prefix + '/test_entity_region.json'
    image_test_path = args.image_path + "/test/"
    model_path = args.model_dir + args.model + args.task+"dev" + '.pt'
    model_path = args.model_dir+ "FGCLNetworktripletbest_test_f10.5340314136125656.pt"
    print(model_path)

    model = torch.load(model_path).to(args.device)


    #
    # print(model)
    # model.eval()

    instances_test = instances_test = load_data_instances(text_test_path,image_test_path, args, clip_args )
    testset = DataIterator(instances_test , args)
    eval(model, testset, args,print_result=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--prefix', type=str, default="./no_none_unified_tags_txt/spacy_result/",
                        help='dataset and embedding path prefix')

    parser.add_argument('--image_path', type=str, default="./img_org/",
                        help='dataset and embedding path prefix')

    parser.add_argument('--model_dir', type=str, default="savemodel/",
                        help='model path prefix')
    parser.add_argument('--task', type=str, default="triplet", choices=["pair", "triplet"],
                        help='option: pair, triplet')
    parser.add_argument('--mode', type=str, default="pre_train", choices=["train", "test","pre_train"],
                        help='option: train, test')
    parser.add_argument('--max_sequence_len', type=int, default=60,
                        help='max length of a sentence')

    parser.add_argument('--device', type=str, default="cuda",
                        help='gpu or cpu')

    parser.add_argument('--bert_model_path', type=str,
                        default="bert-base-uncased",
                        help='pretrained bert model path')

    parser.add_argument('--roberta_model_path', type=str,
                        default="roberta-base",
                        help='pretrained bert model path')

    parser.add_argument('--bert_tokenizer_path', type=str,
                        default="bert-base-uncased",
                        help='pretrained bert tokenizer path')


    parser.add_argument('--roberta_tokenizer_path', type=str,
                        default="roberta-base",
                        help='pretrained bert tokenizer path')


    parser.add_argument('--image_pretrained_path', type=str,
                        default="openai/clip-vit-base-patch32",
                        help='pretrained image model pah')

    parser.add_argument('--cross_attention_heads', type=int, default=4,
                        help='attribute transformer attention')

    parser.add_argument('--bert_feature_dim', type=int, default=768,
                        help='dimension of pretrained bert feature')
    parser.add_argument('--hidden_dim', type=int, default=192,
                        help='dimension of pretrained bert feature')

    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=80,
                        help='training epoch number')
    parser.add_argument('--nhops', type=int, default=2,
                        help='training epoch number')

    parser.add_argument('--class_num', type=int, default=26,
                        help='label number')
    parser.add_argument('--alpha_align_loss', type=float, default=0.001,
                        help='alpha')
    parser.add_argument('--alpha_entity_loss', type=float, default=2,
                        help='alpha')
    parser.add_argument('--lr', type=float, default=2e-4,
                        )

    parser.add_argument('--trans_image_dro', type=float, default=0.4,
                        help='')

    parser.add_argument('--attention_heads', type=int, default=12,
                        help='attribute transformer attention')


    parser.add_argument('--seed', type=int, default=97)

    parser.add_argument('--decline', type=int, default=30, help="number of epochs to decline")
    parser.add_argument('--tau', type=float, default=0.12)

    parser.add_argument('--model', type=str, default="FGCLNetwork")

    parser.add_argument('--pretrain_data', type=str, default="./pre_train_data/",
                        help='')
    parser.add_argument('--pretrain_image', type=str, default="./pre_train_data/dataset/image",
                        help='')





    args = parser.parse_args()
    setup_seed(args.seed)
    clip_args = CLIPConfig.from_pretrained(args.image_pretrained_path)
    if args.mode == 'train':
        train(args,clip_args)
    if args.mode =="pre_train":
        args.lr = float(1e-5)
        pre_train(args, clip_args)
    else:
        test(args,clip_args)
