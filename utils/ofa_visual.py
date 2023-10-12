from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import cv2
import jsonlines
ofa_pipe = pipeline(
    Tasks.visual_grounding,
    model='damo/ofa_visual-grounding_refcoco_large_en')
def process(data_name):
    text_path = "../no_none_unified_tags_txt/{0}_entity.json".format(data_name)
    image_path = "../img_org/{0}/".format(data_name)
    new_data = []
    with open(text_path,"r+") as f:
        for item in jsonlines.Reader(f):
            print(item)
            image_id = image_path+item["img_id"]
            print(image_id)
            # text = 'Barack Hussein Obama'
            # input = {'image': image_id, 'text': text}
            # results = ofa_pipe(input)
            image_id= 'visual_grounding.png'
            text = 'a blue turtle-like pokemon with round head'
            input = {'image': image_id, 'text': text}
            results = ofa_pipe(input)
            print(results)

            img = cv2.imread(image_id)
            # 画矩形框 距离靠左靠上的位置
            result = results["boxes"][0]
            print(result)
            pt1 = (int(result[0]), int(result[1]))  # 左边，上边   #数1 ， 数2
            pt2 = (int(result[2]), int(result[3]))  # 右边，下边  #数1+数3，数2+数4
            print(pt1)
            print(pt2)
            print(img.shape)
            # exit()
            cv2.rectangle(img, pt1, pt2, (255,0,0), 2)

            cv2.imshow('src',img)
            cv2.waitKey()
            cv2.imwrite('22.jpg', img)
            exit()




if __name__ == '__main__':
    path = "train"
    # val_path = "../no_none_unified_tags_txt/train_entity.json"
    # test_path = "../no_none_unified_tags_txt/test.json"
    process(path)


