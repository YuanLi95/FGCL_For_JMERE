from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import cv2
import jsonlines
ofa_pipe = pipeline(
    Tasks.visual_grounding,
    model='damo/ofa_visual-grounding_refcoco_large_en')
def process(data_name):
    text_path = "../no_none_unified_tags_txt/{0}_entity_spacy_trf.json".format(data_name)
    image_path = "../img_org/{0}/".format(data_name)
    new_data = []

    with open(text_path,"r+") as f:
        i=0
        for item in jsonlines.Reader(f):
            print(i)
            i+=1
            image_id = image_path+item["img_id"]
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
            new_data.append({'token':item["token"],'img_id':item['img_id'],"label_list":item["label_list"],"entity_region":entity_region})

    with jsonlines.open("../no_none_unified_tags_txt/{0}_entity_region_spacy_trf.json".format(data_name), "w") as f:
        for index,text in enumerate(new_data):
            print(index)
            f.write(text)





if __name__ == '__main__':
    path_list = ["val","test","train"]
    for path in path_list:
        process(path)


