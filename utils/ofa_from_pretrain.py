from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import cv2
import jsonlines
ofa_pipe = pipeline(
    Tasks.visual_grounding,
    model='damo/ofa_visual-grounding_refcoco_large_en')
def process():
    text_path = "../pre_train_data/pretrain_data_withchat.json"
    image_path = "../pre_train_data/dataset/image/"
    new_data = []
    with jsonlines.open("../pre_train_data/pretrained_entity_region.json", "a") as f1:
        with open(text_path,"r+",encoding="utf-8") as f:
            i=0
            for item in jsonlines.Reader(f):
                print(i)
                i+=1
                if i<22719:
                    continue
                try:
                    print(item["img_id"])
                    if item["img_id"].endswith('jpg'):
                        new_img_id = item["img_id"]
                    else:
                        new_img_id = item["img_id"] + '.jpg'

                    image_id = image_path+new_img_id

                    chat_list = item["chat_entity"]
                    entity_region = []
                    for name,value in chat_list.items():

                            for entity in value:

                                if len(entity.strip())==0:
                                    continue

                                text = "a {0} named {1}".format(name,entity)
                                # print(text)
                                input = {'image': image_id, 'text': text}
                                results = ofa_pipe(input)
                                result = results["boxes"][0]
                                # print(result)
                                entity_region.append([entity,name,result])
                    a = {'token':item["token"],'img_id':new_img_id,"entity_region":entity_region}
                    f1.write(a)
                except:
                    print("Error")






if __name__ == '__main__':
    process()


