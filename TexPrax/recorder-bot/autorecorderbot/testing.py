
from config import Config
from intelligence import SentenceClassPredictor, TokenClassPredictor
import torch


#dieser Path muss angepasst werden zu deiner config datei - diese sollte sich im recorder bot ordner befinden
config_path = "/Users/maira/Desktop/Arbeit/TexPrax/recorder-bot/config.yaml"
config = Config(config_path)


# Test Sentence Class Prediction
SCP = SentenceClassPredictor(config.sequence_model_path)

message = 'Ich habe ein Problem mit der CNC Fräse'

tokenized = SCP.tokenizer(message, truncation=True, padding=True, return_tensors='pt')
input_ids = tokenized['input_ids']
result = torch.argmax(SCP.model(input_ids).logits, 1)
predict = end = SCP.id2label[str(result.item())]

print(tokenized)
print('Prediction: ')
print(predict)




# Test Token Class Prediction
#in welche tokens wird der Satz aufgesplitted
TCP = TokenClassPredictor(config.token_model_path)
print(TCP.predict(message))


