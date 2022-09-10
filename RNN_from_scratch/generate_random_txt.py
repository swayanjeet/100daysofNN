import json
import torch
import pickle
import numpy as np
vocab = None
with open("vocab.txt","r", encoding="utf-8") as f1:
    vocab = json.load(f1)
print(vocab)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.RNN(input_size=len(vocab), hidden_size=100, num_layers=2, nonlinearity="relu", bias=False, batch_first=True, dropout=0.5, bidirectional=False)
        self.fcn = torch.nn.Sequential(
            torch.nn.Linear(in_features=100, out_features=len(vocab), bias=True)
        )
    
    def forward(self, x):
        output = self.rnn(x)
        return self.fcn(output[0])

model = Model()
model.load_state_dict(torch.load("rnn_model_latest.pt"))
model.eval()
with open("data","rb") as f1:
    data = pickle.load(f1)
time_step_len = 4
len_dataset = 100000
data_ids_p = [c for c in data[len_dataset-600:len_dataset-600+10]]
print("Initial Data ids",data_ids_p)
number_of_chars_to_generate = 50
generated_chars = data_ids_p
for m in range(0, number_of_chars_to_generate):
    data_ids_p = data_ids_p[len(data_ids_p)-time_step_len:len(data_ids_p)]
    print("After updation",data_ids_p)
    data_ids_vectors = np.zeros((len_dataset,len(vocab)))
    for i,data in enumerate(data_ids_p):
        data_ids_vectors[i][vocab[data]] = 1
    array_in = np.empty((1,time_step_len,len(vocab)))
    for j in range(0,1):
        for i in range(0,time_step_len):
            array_in[j][i] = data_ids_vectors[j+i]

    input_chars = torch.from_numpy(array_in)
    op = model(input_chars.float())
    output_indices = torch.argmax(op[:,(time_step_len-1),:], dim=1)
    for key, value in vocab.items():
        if value == output_indices.item():
            print("Next Key is {}".format(key))
            data_ids_p.append(key)
            generated_chars.append(key)
print("".join(generated_chars))
