import numpy as np
import aisuite as ai
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from classification.classification_models.vit import ClassificationModel, ModelConfig
from classification.classification_models.vit import ImageWoofDataset
from classification.classification_metrics.metrics import ClassificationMetrics
from pathlib import Path
import albumentations as A
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
import json

class ClassificationErrorAnalysis(pl.LightningModule):
    def __init__(self, model:nn.Module, dataloader: DataLoader):
        super().__init__()
        self.model = model
        self.dataloader = dataloader
    def get_statistics(self):
        # TODO: implement parallel processing for this
        all_preds = []
        all_probs = []
        all_labels = []

        for batch in tqdm(self.dataloader):
            x, y = batch
            y_hat = self.model(x)
            probs = torch.softmax(y_hat, dim=1)
            pred_classes = torch.argmax(probs, dim=1)

            all_probs.append(probs)
            all_preds.append(pred_classes)
            all_labels.append(y)
        
        all_probs = torch.cat(all_probs, dim=0)
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        self.aucroc = ClassificationMetrics.get_aucroc(all_labels, all_probs)
        self.f1 = ClassificationMetrics.get_f1(all_labels, all_preds)
        self.accuracy = ClassificationMetrics.get_accuracy(all_labels, all_preds)
        self.precision = ClassificationMetrics.get_precision(all_labels, all_preds)
        self.statistics = {
            "aucroc": self.aucroc,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "precision": self.precision
        }
        with open("statistics.json", "w") as f:
            json.dump(self.statistics, f)
    

class ClassificationResearchAgent(ClassificationErrorAnalysis):
    def __init__(self, 
                 user_prompt: str,  
                 trained_model: nn.Module,
                 dataloader: DataLoader,
                 researcher_model_name:"ollama:gemini-3-flash-preview"
    ):
    
        super().__init__(trained_model, dataloader)
        self.user_prompt = user_prompt
        self.ai = ai.Client()
        self.system_prompt = f"""
        You are an experienced computer vision practitioner. 
        You are given a set of statistics and a user prompt. 
        You need to analyze the statistics and recommend a set of changes to the user prompt to improve the model's performance.
        """
    def analyze_and_recommend(self):
        # messages = [
        #     {"role": "system", "content": self.system_prompt},
        #     {"role": "user", "content": self.user_prompt}
        # ]
        self.get_statistics()
        
        # results = self.ai.chat.completions.create(
        #     model=self.model_name,
        #     messages=messages,
        #     max_tokens=1000,
        #     temperature=0.2,
        # )
        
        
        # return results.choices[0].message.content

if __name__ == "__main__":
    from lightning.pytorch.callbacks import ModelCheckpoint
    trained_model = ClassificationModel.load_from_checkpoint(
        "classification/classification_models/lightning_logs/version_4/checkpoints/tinynet-epoch=10-val_acc=0.8400.ckpt"
    )
    trained_model.eval()

    val_path = Path('/Users/jeremyong/Desktop/research_agent/dataset/imagewoof-160/val')
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

    research_agent = ClassificationResearchAgent(
        user_prompt="What is the latest research in computer vision related to multimodal object detection?",
        trained_model=trained_model,
        dataloader=DataLoader(ImageWoofDataset(val_path, transform=val_transform), batch_size=2,num_workers=4, shuffle=False),
        researcher_model_name="ollama:gemini-3-flash-preview"
    )

    research_agent.analyze_and_recommend()
