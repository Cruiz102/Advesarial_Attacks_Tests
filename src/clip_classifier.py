from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
from transformers.modeling_outputs import ImageClassifierOutput
from typing import List
import torch 
import torch.nn as nn

class clip_classifier(nn.Module):
    def __init__(self,labels_name: List[str], clip_model_name: str = None, device = torch.device("cuda"))-> torch.Tensor:
        super().__init__()
        self.labels_name = labels_name
        self.device = device
        if clip_model_name is not None:
            self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        else:
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        self.inputs_ids = self.clip_processor(text=self.labels_name, return_tensors="pt", padding=True)["input_ids"]
        self.inputs_ids = self.inputs_ids.to(device)


    def forward(self, images)-> torch.Tensor:
        images = images.to(self.device)
        inputs = {"pixel_values": images, "input_ids" :self.inputs_ids}
        outputs = self.clip_model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image
        result = ImageClassifierOutput(logits=probs)
        return result