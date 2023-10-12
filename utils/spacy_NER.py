import time

import openai
import json
import  jsonlines

import spacy

idx = {"person":"person","law":"person","norp":"organization",
       "fac":"location","org":"organization","gpe":"location","loc":"location","event":"miscellaneous","work_of_art":"miscellaneous"}

#加载预训练模型

# nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("en_core_web_trf")

def process(data_set_path):


    file = json.load(open(data_set_path))

    with jsonlines.open("../no_none_unified_tags_txt/test_entity_spacy_trf.json", "w") as f:
        for index, text in enumerate(file):
            person_list = []
            org_list = []
            loc_list = []
            mis_list = []

            test_text = text["token"]

            test_text =' '.join(test_text)
            print(test_text)
            doc = nlp(test_text)
            for ent in doc.ents:
                label = ent.label_.lower()
                if label not in idx:
                    continue
                convert_label = idx[label]
                if convert_label == "person":
                    person_list.append(ent.text)
                elif convert_label == "organization":
                    org_list.append(ent.text)
                elif convert_label == "location":
                    loc_list.append(ent.text)
                elif convert_label == "miscellaneous":
                    mis_list.append(ent.text)

            entity_dict = {"person": person_list, "location": loc_list, "organization": org_list, "miscellaneous":mis_list}
            text["chat_entity"] = entity_dict

            f.write(text)


if __name__ == '__main__':
    train_path = "../no_none_unified_tags_txt/train.json"
    val_path = "../no_none_unified_tags_txt/val.json"
    test_path = "../no_none_unified_tags_txt/test.json"
    process(test_path)






