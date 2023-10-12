import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from transformers import BertModel,RobertaModel
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer,CLIPConfig
from .Att_transformer import  Crooss_attention

# from .crfasrnn.crfrnn import CrfRnn
# from crfasrnn.crfrnn import CrfRnn


class FGCLNetwork(torch.nn.Module):




    def __init__(self, args,clip_args):
        super(FGCLNetwork, self).__init__()
        self.args = args
        self.hidden_dim= args.hidden_dim
        # self.bert = RobertaModel.from_pretrained(args.roberta_model_path, return_dict=False)
        self.bert = BertModel.from_pretrained(args.bert_model_path,return_dict = False)
        self.pseudo_emb_layer = nn.Embedding(5,args.hidden_dim)
        self.clip_image = CLIPModel.from_pretrained(args.image_pretrained_path)

        self.trans_image = nn.Sequential(torch.nn.Linear(clip_args.vision_config.hidden_size, 32), nn.ReLU(),
                                       torch.nn.Linear(32, int(self.hidden_dim)))
        # self.trans_token  = nn.Sequential(torch.nn.Linear(args.bert_feature_dim, 32), nn.ReLU(),
        #               torch.nn.Linear(32, int(self.hidden_dim/2)))

        self.trans_token = nn.Sequential(self.Linear(args.bert_feature_dim,int(self.hidden_dim)), nn.ReLU(),nn.Dropout(0.2))
        self.tau = args.tau
        self.imtotext_cross_attention = Crooss_attention(args.cross_attention_heads, self.hidden_dim, self.hidden_dim,
                                                         self.hidden_dim)
        self.text_attention = Crooss_attention(args.cross_attention_heads, self.hidden_dim, self.hidden_dim,
                                                         self.hidden_dim)
        self.image_attention = Crooss_attention(args.cross_attention_heads, self.hidden_dim, self.hidden_dim,
                                                         self.hidden_dim)
        self.device = args.device

        self.dropout_all = nn.Dropout(0.2)
        self.gru = nn.GRU(self.hidden_dim*2,self.hidden_dim*2,batch_first=True)
        self.drop_out = nn.Dropout(0.2)
        self.linear_out = torch.nn.Linear(self.hidden_dim*2, args.class_num)
        self.feature_linear = torch.nn.Linear(args.hidden_dim* 2 + args.class_num * 3, args.hidden_dim * 2)
        self.feature_linear_table = torch.nn.Linear(args.hidden_dim* 2 + args.class_num * 1, args.hidden_dim * 2)

        self.entity_linear_out = torch.nn.Linear(int(self.hidden_dim),5)
        self.stack_logistic_linear = torch.nn.Linear(args.nhops+1,1,bias=False)
        self.gate = nn.GRU(args.hidden_dim, args.hidden_dim, batch_first=True)
        # self.crfrnn = CrfRnn(args.class_num)
    def Linear(self,inputdim, outputdim, bias=True, uniform=True):
        linear = nn.Linear(inputdim, outputdim, bias)
        if uniform:
            nn.init.xavier_uniform_(linear.weight)
        else:
            nn.init.xavier_normal_(linear.weight)
        if bias:
            nn.init.constant_(linear.bias, 0.0)
        return linear
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):  # cal cosine simility
        z1 = F.normalize(z1, dim=2)
        z2 = F.normalize(z2, dim=2)
        return torch.bmm(z1, z2.transpose(1,2))
    def FG_contrastive_loss(self, x_token,x_image,token_masks,contras_image_tags, contras_imagetotext_tags,contras_text_tags,
                                          pos_image_idx,pos_imagetotext_idx,pos_text_idx):

        f = lambda x: torch.exp(x / self.tau)  # f: e^(f(z1,z2)/t)
        # exit()
        bs, seq,dh = x_token.shape
        x_token = x_token * token_masks.unsqueeze(-1).expand(bs, seq,dh)
        _,image_len, dh= x_image.shape
        pos_text_idx = pos_text_idx.unsqueeze(-1).expand(bs, seq,seq)
        token_sim_all_org = self.sim(x_token, x_token)*pos_text_idx
        token_sim_all = f(token_sim_all_org)
        token_sim_positive = f(token_sim_all_org*contras_text_tags)


        pos_image_idx = pos_image_idx.unsqueeze(-1).expand(bs, image_len, image_len)
        image_sim_all_org = self.sim(x_image, x_image) * pos_image_idx
        image_sim_all = f(image_sim_all_org)

        image_sim_positive = f(image_sim_all_org * contras_image_tags)


        pos_imagetotext_idx = pos_imagetotext_idx.unsqueeze(-1).expand(bs, image_len, seq)
        between_sim_imagetotext_org = self.sim(x_image,x_token) * pos_imagetotext_idx
        between_sim_imagetotext =  f(between_sim_imagetotext_org)
        between_sim_imagetotext_positive =  f(between_sim_imagetotext_org *contras_imagetotext_tags)

        # print(between_sim_imagetotext_positive.shape)


        between_sim_texttoimage_org = self.sim(x_token, x_image) * (pos_imagetotext_idx.transpose(1,2))
        between_sim_texttoimage =f(between_sim_texttoimage_org)
        between_sim_texttoimage_positive = f(between_sim_texttoimage_org*contras_imagetotext_tags.transpose(1,2))

        # print(between_sim_texttoimage_positive.shape)
        # exit()

        # weighted_imagetotext = f(torch.mul((self.sim(x_image, x_token) * pos_imagetotext_idx), (self.sim(x_image, x_token) * pos_imagetotext_idx).diagonal(dim1=-2,dim2=-1).unsqueeze(dim=-1)))
        # weighted_texttoimage = f(torch.mul((self.sim(x_token, x_image) * (pos_imagetotext_idx.transpose(1,2))), (self.sim(x_token, x_image) * (pos_imagetotext_idx.transpose(1,2))).diagonal(dim1=-2,dim2=-1).unsqueeze(dim=-1)))

        text_logit = (token_sim_positive.sum(1) - token_sim_positive .diagonal(dim1=-2,dim2=-1)) / token_sim_all.sum(1)
        image_logit = (image_sim_positive.sum(1) - image_sim_positive.diagonal(dim1=-2,dim2=-1)) / image_sim_all.sum(1)

        imagetotext_logit = 0.5*(between_sim_imagetotext_positive.sum(1))/between_sim_imagetotext.sum(1)
        # print(between_sim_imagetotext_positive.shape)
        # print("1111111111")
        # print(imagetotext_logit.shape)
        # print(between_sim_texttoimage_positive.shape)
        # print(between_sim_texttoimage.shape)
        # print(between_sim_texttoimage_positive.sum(2).shape)
        # print(between_sim_texttoimage_positive.diagonal(dim1=-1,dim2=-2).shape)
        # print(between_sim_texttoimage_positive.diagonal(dim1=-2, dim2=-1).shape)
        # exit()
        # print(token_sim_all.sum(1))
        texttoimage_logit = 0.5*(between_sim_texttoimage_positive.sum(1))/between_sim_texttoimage.sum(1)
        # print(text_logit)
        # print(image_logit)
        # print(imagetotext_logit)
        # print(texttoimage_logit)
        # print("1111111111111111")
        text_left = -torch.log(text_logit)
        image_left = -torch.log(image_logit)
        imagetotext_left =  -torch.log(imagetotext_logit)
        texttoimage_left = -torch.log(texttoimage_logit)
        # print(text_left)
        # print(image_left)
        # print(imagetotext_left)
        # print(texttoimage_left)
        text_all = text_left + imagetotext_left
        image_all = image_left + texttoimage_left
        ret_text = text_all.mean(dim=1, keepdim=True)

        ret_image = image_all.mean(dim=1, keepdim=True)
        return ret_text+ret_image

    def FG_cross_loss(self, x_token,x_image,token_masks,contras_image_tags, contras_imagetotext_tags,contras_text_tags,
                                          pos_image_idx,pos_imagetotext_idx,pos_text_idx):
        bs, seq, dh = x_token.shape
        x_token = x_token * token_masks.unsqueeze(-1).expand(bs, seq, dh)
        _, image_len, dh = x_image.shape
        pos_text_idx = pos_text_idx.unsqueeze(-1).expand(bs, seq, seq)
        pos_image_idx = pos_image_idx.unsqueeze(-1).expand(bs, image_len, image_len)
        pos_imagetotext_idx = pos_imagetotext_idx.unsqueeze(-1).expand(bs, image_len, seq)
        # print(x_token.shape)
        # print(x_image.shape)
        # print()
        #
        # loss_imagetotext = F.cross_entropy(preds_flatten, tags_flatten, ignore_index=-1)



    def multi_hops(self, features, mask, k):
        '''generate mask'''
        max_length = features.shape[1]
        mask = mask[:, :max_length]
        mask_a = mask.unsqueeze(1).expand([-1, max_length, -1])
        mask_b = mask.unsqueeze(2).expand([-1, -1, max_length])
        mask = mask_a * mask_b
        mask = torch.triu(mask).unsqueeze(3).expand([-1, -1, -1, self.args.class_num])

        '''save all logits'''
        logits_list = []
        logits = self.linear_out(features)
        logits_list.append(logits)

        for i in range(k):
            # probs = torch.softmax(logits, dim=3)
            probs = logits
            logits = probs * mask

            logits_a = torch.max(logits, dim=1)[0]
            logits_b = torch.max(logits, dim=2)[0]
            logits = torch.cat([logits_a.unsqueeze(3), logits_b.unsqueeze(3)], dim=3)
            logits = torch.max(logits, dim=3)[0]

            logits = logits.unsqueeze(2).expand([-1, -1, max_length, -1])
            logits_T = logits.transpose(1, 2)
            logits = torch.cat([logits, logits_T], dim=3)

            new_features = torch.cat([features, logits, probs], dim=3)

            features = self.feature_linear(new_features)
            logits = self.linear_out(features)
            logits_list.append(logits)
        return logits_list

    def tag_decoding(self, features, mask, k):
        # print(features.shape)
        bacth_size, max_length,max_length,dim = features.shape
        mask = mask[:, :max_length]
        mask_a = mask.unsqueeze(1).expand([-1, max_length, -1])
        mask_b = mask.unsqueeze(2).expand([-1, -1, max_length])
        mask = mask_a * mask_b

        '''save all logits'''
        logits_list = []
        logits = self.linear_out(features)
        logits_list.append(logits)
        left_feature = torch.zeros(features.shape).to(self.args.device)
        right_feature = torch.zeros(features.shape).to(self.args.device)
        up_feature = torch.zeros(features.shape).to(self.args.device)
        down_feature = torch.zeros(features.shape).to(self.args.device)

        for i in range(k):
            # probs = torch.softmax(logits, dim=3)
            probs = logits
            # features_add = torch.cat([features, probs], dim=3)
            #
            # new_features = self.feature_linear_table(features_add)
            left_feature[:, 1:, :, :] = features[:, :-1, :, :]
            right_feature[:, :-1, :, :] = features[:, 1:, :, :]
            up_feature[:, :, 1:, :] = features[:, :, :-1, :]
            down_feature[:, :, :-1, :] = features[:, :, 1:, :]

            # left_feature[:, 1:, :, :] = new_features[:, :-1, :, :]
            # right_feature[:, :-1, :, :] = new_features[:, 1:, :, :]
            # up_feature[:, :, 1:, :] = new_features[:, :, :-1, :]
            # down_feature[:, :, :-1, :] = new_features[:, :, 1:, :]


            old_feature = features
            features_add = torch.cat([features, probs], dim=3)
            # features_add = torch.cat([features, logits, probs], dim=3)
            #
            new_features = self.feature_linear_table(features_add)

            features_other = torch.cat([new_features.unsqueeze(-2), old_feature.unsqueeze(-2),left_feature.unsqueeze(-2),
                                        right_feature.unsqueeze(-2),up_feature.unsqueeze(-2),down_feature.unsqueeze(-2)], dim=-2)
            # exit()
            # features_other = torch.cat([new_features.unsqueeze(-2),left_feature.unsqueeze(-2),
            #                             right_feature.unsqueeze(-2),up_feature.unsqueeze(-2),down_feature.unsqueeze(-2)], dim=-2)

            features_other = features_other.view( -1,6,dim)
            # features_other = features_other.view(-1, 5, dim)


            _, features  = self.gru(features_other)

            features = features.squeeze(1).view(bacth_size,max_length,max_length,dim)

            logits = self.linear_out(features)

            logits_list.append(logits)

        logits_list = torch.stack(logits_list, dim=-2)
        # print(logits_list.shape)
        return logits_list

    def Pretrain(self,bert_tokens, lengths, token_masks, sens_lens, image_input, \
                contras_image_tags, contras_imagetotext_tags, contras_text_tags,
                pos_image_idx, pos_imagetotext_idx, pos_text_idx,pseudo_label_idx):
        bs,token_seq,= bert_tokens.shape

        #得到正
        image_masks = torch.ones_like(contras_image_tags).to(self.device)

        cross_attention_mask = token_masks.unsqueeze(2)

        out_bert = self.bert(bert_tokens, token_masks)
        x_token, cls_token = out_bert[0], out_bert[1]
        # print(44444444444444444444444444444444444)

        image_encoding = self.clip_image.get_image_features(pixel_values=image_input,
                                                            return_vision_outputs=True).last_hidden_state
        # print(5555555555555555555555)
        # exit()
        image_representation = image_encoding[:, 1:, :]
        _, image_seq, _ = image_representation.shape
        x_image = self.trans_image(image_representation)
        cross_attention_mask = torch.repeat_interleave(cross_attention_mask, dim=2, repeats=image_seq).double()
        x_token = self.trans_token(x_token)
        pseudo_embedding = self.pseudo_emb_layer(pseudo_label_idx)
        x_token = x_token + pseudo_embedding
        # print(x_token.shape)
        # print(token_masks.shape)
        # exit()
        token_masks_repeat = token_masks.unsqueeze(1).repeat(1, token_seq, 1)

        x_token, _ = self.text_attention(x_token, x_token, x_token, token_masks_repeat)

        x_image, _ = self.image_attention(x_image, x_image, x_image, image_masks)

        # print(pseudo_label_idx)

        # print(x_image.shape)
        # print(x_token.shape)

        # print(x_token.shape)
        loss_1 = self.FG_contrastive_loss(x_token, x_image, token_masks, contras_image_tags, contras_imagetotext_tags,
                                          contras_text_tags,
                                          pos_image_idx, pos_imagetotext_idx, pos_text_idx).mean()

        return  loss_1

    def _get_CFR(self, sentence):
        # sentence = sentence[0]
        sentence = sentence.view(1, -1).float()
        CFR_feature = sentence.T + sentence
        CFR_feature = CFR_feature.expand(3, -1, -1)
        return CFR_feature

    def forward(self,bert_tokens, lengths, token_masks, sens_lens, image_input, \
                contras_image_tags, contras_imagetotext_tags, contras_text_tags,
                pos_image_idx, pos_imagetotext_idx, pos_text_idx,pseudo_label_idx):
        bs,token_seq,= bert_tokens.shape

        #得到正

        image_masks = torch.ones_like(contras_image_tags).to(self.device)

        cross_attention_mask= token_masks.unsqueeze(2)

        out_bert = self.bert(bert_tokens,token_masks)
        x_token,cls_token = out_bert[0],out_bert[1]
        # print(44444444444444444444444444444444444)

        image_encoding = self.clip_image.get_image_features(pixel_values=image_input,return_vision_outputs=True).last_hidden_state
        # print(5555555555555555555555)
        # exit()
        image_representation = image_encoding[:,1:,:]
        _,image_seq,_ = image_representation.shape
        x_image = self.trans_image(image_representation)
        cross_attention_mask = torch.repeat_interleave(cross_attention_mask, dim=2, repeats=image_seq).double()
        x_token = self.trans_token(x_token)
        pseudo_embedding = self.pseudo_emb_layer(pseudo_label_idx)
        x_token = x_token + pseudo_embedding
        token_masks_repeat = token_masks.unsqueeze(1).repeat(1,token_seq,1)

        x_token,_ = self.text_attention(x_token, x_token, x_token, token_masks_repeat)


        x_image,_ = self.image_attention(x_image, x_image, x_image,image_masks)

        # print(pseudo_label_idx)


        # print(x_image.shape)
        # print(x_token.shape)

        # print(x_token.shape)
        loss_1 =self.FG_contrastive_loss(x_token,x_image, token_masks,contras_image_tags, contras_imagetotext_tags,contras_text_tags,
                                          pos_image_idx,pos_imagetotext_idx,pos_text_idx).mean()

        # loss_1 = 0

        imagefortext, _ = self.imtotext_cross_attention(x_token, x_image, x_image, cross_attention_mask)

        # print(torch.cat([imagefortext.unsqueeze(2),x_token.unsqueeze(2)],dim=2).shape)
        gate_input = torch.cat([imagefortext.unsqueeze(2),x_token.unsqueeze(2)],dim=2).reshape(bs*token_seq,-1,self.hidden_dim)
        _, out_features  = self.gate(gate_input)

        out_hidden = out_features .reshape(bs,token_seq,self.hidden_dim)


        pre_entity = self.entity_linear_out(out_hidden)
        final_feature = x_token.unsqueeze(2).expand([-1, -1, token_seq, -1])
        final_feature_T = final_feature.transpose(1, 2)
        features = torch.cat([final_feature, final_feature_T], dim=3)
        # logits = self.multi_hops(features, token_masks, self.args.nhops)
        logits = self.tag_decoding(features, token_masks, self.args.nhops)
        logitc_output = self.stack_logistic_linear(logits.transpose(-1,-2)).squeeze()
        #
        # logitc_output = logitc_output.permute(0, 3, 1, 2)
        # image_feature = self._get_CFR(bert_tokens[0][:lengths[0]])
        # image_feature = image_feature .unsqueeze(0)
        # print(image_feature.shape)
        # print(logitc_output.shape)
        # out_put = self.crfrnn(image_feature, logitc_output)
        # print(out_put.shape)


        # print(logitc_output.shape)
        # logits = logits[-1]
        # print(features.shape)
        # print(features.shape)
        # print(token_pospeech_matrix.shape)
        # print(token_frequency_embed.shape)
        # print(syntax_position_channel.shape)
        # print(features.shape)
        # exit()
        # loss_1=0
        # concat_features = torch.cat([features,token_frequency_embed,token_pospeech_matrix,syntacx_postition_embed], dim=3)
        # print(logits.shape)
        # return logits, loss_1,pre_entity
        return  logitc_output, loss_1,pre_entity



