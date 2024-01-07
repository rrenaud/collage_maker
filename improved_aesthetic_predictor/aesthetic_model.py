# Taken from https://github.com/christophschuhmann/improved-aesthetic-predictor/tree/main 
# which is part of the LAION project.

from PIL import Image
import io

import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import clip

from PIL import Image

class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class AestheticInferenceResult(object):
    def __init__(self, aesthetic_score, embedding):
        self.aesthetic_score = aesthetic_score
        self.embedding = embedding


def cdn(torch_tensor):
    return torch_tensor.cpu().detach().numpy()


class AestheticModel(object):
    def __init__(self, model_path="improved_aesthetic_predictor/sac+logos+ava1-l14-linearMSE.pth"):
        self.aesthetic_model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14        
        s = torch.load(model_path) 
        self.aesthetic_model.load_state_dict(s)

        self.aesthetic_model.to("cuda")
        self.aesthetic_model.eval()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load("ViT-L/14", device=self.device)  #RN50x64   

    def infer(self, image: Image) -> AestheticInferenceResult:
        with torch.no_grad():
            preprocessed_image = self.preprocess(image).unsqueeze(0).to(self.device)
            image_features = self.clip_model.encode_image(preprocessed_image)
            im_emb_arr = normalized(cdn(image_features.cpu()))
            prediction = self.aesthetic_model(torch.from_numpy(im_emb_arr).to(self.device).type(torch.cuda.FloatTensor))
            return AestheticInferenceResult(cdn(prediction), im_emb_arr)


if __name__ == '__main__':
    import sys
    pil_image = Image.open("test.jpg" if len(sys.argv) < 2 else sys.argv[1])
    model = AestheticModel()

    prediction = model.infer(pil_image)

    print( "Aesthetic score predicted by the model:")
    print( prediction.aesthetic_score)

