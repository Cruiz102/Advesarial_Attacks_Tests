from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
from typing import List
import torch 
import torch.nn as nn

class clip_classifier(nn.Module):
    def __init__(self,labels_name: List[str], clip_model_name: str = None)-> torch.Tensor:
        super().__init__()
        self.labels_name = labels_name
        if clip_model_name is not None:
            self.clip_model = CLIPModel.from_pretrained(clip_model_name)
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        else:
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")



    def forward(self, images)-> torch.Tensor:
        inputs = self.clip_processor(text=self.labels_name, images=images, return_tensors="pt", padding=True)
        outputs = self.clip_model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
        return probs