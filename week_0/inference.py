import torch
from model import ColaModel
from data import DataModule

class ColaPredictor:
    def __init__(self, model_path, model_name):
        self.model_path = model_path
        self.model = ColaModel.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.freeze()
        self.processor = DataModule(model_name=model_name)  # Pass the model_name
        self.softmax = torch.nn.Softmax(dim=1)  # Apply softmax along dim=1 (batch dimension)
        self.labels = ["unacceptable", "acceptable"]

    def predict(self, text):
        inference_sample = {"sentence": text}
        processed = self.processor.tokenize_data(inference_sample)
        logits = self.model(
            torch.tensor([processed["input_ids"]]),
            torch.tensor([processed["attention_mask"]]),
        )
        scores = self.softmax(logits)
        predictions = []
        for score, label in zip(scores[0], self.labels):
            predictions.append({"label": label, "score": score.item()})  # Convert score tensor to float
        return predictions

if __name__ == "__main__":
    sentence = "The boy is sitting on a bench"
    model_path = "./models/epoch=0-step=267.ckpt"
    model_name = "google/bert_uncased_L-2_H-128_A-2"  # Specify the correct model name
    predictor = ColaPredictor(model_path, model_name)
    print(predictor.predict(sentence))
