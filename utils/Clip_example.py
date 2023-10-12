from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer,CLIPConfig
#
# A = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")
# print(A)
# exit()
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14",)
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_inputs = tokenizer(["a photo of a cat a photo of a cat a photo of a cat a photo of a cat a photo of a cat a photo of a cat ", "a photo of a dog"], padding=True, return_tensors="pt")
text_hidden_state = model.get_text_features(**text_inputs,return_text_outputs=True)
print(text_hidden_state)

inputs = processor(images=image, return_tensors="pt")
print(inputs.pixel_values.shape)
# exit()
image_hidden_state = model.get_image_features(**inputs,return_vision_outputs=True)

print(text_hidden_state.last_hidden_state.shape)
exit()
print("11111111111111111")
print(image_hidden_state.last_hidden_state.shape)

# text_outputs = outputs.text_outputs
# print(text_outputs)
# vision_outputs = outputs.vision_outputs
# print(vision_outputs)
# exit()
# logits_per_image = outputs.logits_per_image # this is the image-text similarity score
# probs = logits_per_image.softmax(dim=1)
# print(probs)