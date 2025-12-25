#!/usr/bin/env python3
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, "/app")

MODEL_PATH = "/gateway_instance/enrollment/speaker_classifier.pth"


class NeuralClassifier(nn.Module):
    def __init__(self):
        super(NeuralClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(192, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.network(x)


def main(filepath):
    print(f"Debug analyzing: {filepath}")

    # 1. Load Model
    device = torch.device("cpu")
    model = NeuralClassifier().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 2. Get Embedding (TitaNet)
    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    titanet = EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
    titanet = titanet.to("cpu").eval()

    # Process
    try:
        if hasattr(titanet, "get_embedding"):
            emb = titanet.get_embedding(filepath)
        else:
            emb, _ = titanet.infer_file(filepath)
            emb = emb[0]

        if hasattr(emb, "cpu"):
            emb = emb.cpu().numpy()
        emb = emb.flatten()
        emb = emb / np.linalg.norm(emb)

        # 3. Classify
        with torch.no_grad():
            t_emb = torch.FloatTensor(emb).unsqueeze(0).to(device)
            logits = model(t_emb)
            probs = F.softmax(logits, dim=1).numpy()[0]

        print(f"   P(Other):  {probs[0]:.4f}")
        print(f"   P(Pruitt): {probs[1]:.4f}")
        print(f"   P(Ericah): {probs[2]:.4f}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
