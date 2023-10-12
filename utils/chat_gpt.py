import time

import openai
import json
import  jsonlines


def process(data_set_path):
    x_1 = ['San', 'Diego', 'Padres', ':', 'Reyes', ',', 'Paddack', ',', 'Radke', 'Nominated', 'For', 'MiLB', 'Awards', '#', 'Padres']
    x_2 = ['RT', '@ManuelDuarte24', ':', 'The', 'scene', 'between', 'Luke', 'and', 'Yoda', 'in', 'The', 'Last', 'Jedi', 'is', 'a', 'top', '5', 'scene', 'in', 'the', 'Star', 'Wars', 'saga']
    x_3 =  ['RT', '@XSovietNews', ':', 'Russia', "'s", 'Foreign', 'Ministry', 'spokeswoman', 'Maria', 'Zakharova', 'captioned', 'this', '"', 'With', 'my', 'colleague', 'from', 'work', '.', '"']
    x_1 = " ".join(x_1)
    x_2 = " ".join(x_2)
    x_3 = " ".join(x_3)
    Pro = "Q: Please identify Person, Location, Organization and Miscellaneous entity from the text \n"
    Q_1 =  Pro+"Text: "+ x_1+"\n"
    A_1 = "###Person: Radke, Reyes, Paddack \n###Location: San Diego Padre \n###Organization: \n###Miscellaneous:"
    Prp_1 =Q_1+A_1+"\n"
    # print(Prp_1)
    # exit()

    Q_2 = Pro+ "Text: "+x_2+"\n"
    A_2 = "###Person: Luke, Yoda \n###Location: \n###Organization: \n###Miscellaneous:Last Jedi,Star Wars"
    Prp_2 =Q_2+A_2+"\n"

    Q_3 = Pro+ "Text: "+x_3+"\n"
    A_3 = "###Person: Maria Zakharova \n###Location: Russia \n###Organization: RT @XSovietNews, Foreign Ministry \n###Miscellaneous:"
    Prp_3 =Q_3+A_3+"\n"

    Prp_all= Prp_1+Prp_2+Prp_3
    # Prp_all= Prp_1



    openai.api_key = "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"
    file = json.load(open(data_set_path))

    with jsonlines.open("../no_none_unified_tags_txt/test_entity_other.json", "w") as f:
        for index, text in enumerate(file):
            print(index)
            # if index<399:
            #     continue
            try:
                test_text = text["token"]
                test_text = " ".join(test_text)
                Q_3 = Pro + "Text: " + test_text + "\n"
                Prp_new = Prp_all + Q_3
                # print(Prp_new)
                completion = openai.ChatCompletion.create(
                  model="gpt-3.5-turbo",
                  messages=[
                      {"role": "user", "content": Prp_new }
                            ]
                )
                # print(Prp_new)
                messge_content = completion["choices"][0]["message"]["content"]
                print(messge_content)
                # exit()
                entity_dict  = {}
                for every_line in messge_content.split("\n"):
                    # print(every_line)
                    header,value = every_line.split(":",maxsplit = 1)
                    header_new= header[3:].lower()
                    if len(value)>0:
                        value_list=[i.strip() for i in value.split(",")]
                    entity_dict[header_new] = value_list
                text["chat_entity"] = entity_dict
                print(text)
                f.write(text)
                time.sleep(4)
            except:
                try:
                    print("请休息二会了")
                    print(index)
                    time.sleep(30)
                    test_text = text["token"]
                    test_text = " ".join(test_text)
                    Q_3 = Pro + "Text: " + test_text + "\n"
                    Prp_new = Prp_all + Q_3
                    # print(Prp_new)
                    completion = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "user", "content": Prp_new}
                        ]
                    )

                    messge_content = completion["choices"][0]["message"]["content"]
                    entity_dict = {}
                    for every_line in messge_content.split("\n"):
                        # print(every_line)
                        header, value = every_line.split(":",maxsplit = 1)
                        header_new = header[3:].lower()
                        if len(value) > 0:
                            value_list = [i.strip() for i in value.split(",")]
                        entity_dict[header_new] = value_list
                    text["chat_entity"] = entity_dict
                    f.write(text)
                except:
                    print("请休息两会了")
                    print(index)
                    time.sleep(60)
                    test_text = text["token"]
                    test_text = " ".join(test_text)
                    Q_3 = Pro + "Text: " + test_text + "\n"
                    Prp_new = Prp_all + Q_3
                    # print(Prp_new)
                    completion = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "user", "content": Prp_new}
                        ]
                    )

                    messge_content = completion["choices"][0]["message"]["content"]
                    entity_dict = {}
                    for every_line in messge_content.split("\n"):
                        # print(every_line)
                        header, value = every_line.split(":",maxsplit = 1)
                        header_new = header[3:].lower()
                        if len(value) > 0:
                            value_list = [i.strip() for i in value.split(",")]
                        entity_dict[header_new] = value_list
                    text["chat_entity"] = entity_dict
                    f.write(text)


if __name__ == '__main__':
    train_path = "../no_none_unified_tags_txt/train.json"
    val_path = "../no_none_unified_tags_txt/val.json"
    test_path = "../no_none_unified_tags_txt/test.json"
    process(test_path)






