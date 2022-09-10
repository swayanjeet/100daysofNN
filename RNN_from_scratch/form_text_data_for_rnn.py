import os
import glob
import pickle
import numpy as np

data = []
# chars = None
# for file_ in glob.glob("jane_austen.txt"):
#     with open(file_) as f1:
#         char_set = []
#         for line in f1.read().split("\n"):
#             for chars in line:
#                 char_set.append(chars)
#         data = data+char_set

# print(data[0:4])
# # # pickle.dump(data)
# with open("data","wb") as f1:
#     data = pickle.dump(data, f1)
with open("data","rb") as f1:
    data = pickle.load(f1)
from torchtext.data.functional import simple_space_split
from torchtext.data.functional import numericalize_tokens_from_iterator
import torch.nn.functional
from torch.utils.data import DataLoader, TensorDataset
import json
len_dataset = 100000
test_train_split = 0.1
data_ids_p = [c for c in data[0:len_dataset]]
set_ = set(data_ids_p)
vocab = dict(zip(set_,[i for i in range(len(set_))]))
print(vocab)
with open("vocab.txt","w",encoding='utf8') as f1:
    json.dump(vocab,f1,ensure_ascii=False)
data_ids_vectors = np.zeros((len_dataset,len(set_)))
for i,data in enumerate(data_ids_p):
    data_ids_vectors[i][vocab[data]] = 1
print(data_ids_vectors[0])
print(data_ids_p[0])
time_step_len = 100
array_in = np.empty((len_dataset-time_step_len,time_step_len,len(set_)))
array_op = np.empty((len_dataset-time_step_len,len(set_)))
for j in range(0,len_dataset-time_step_len):
    for i in range(0,time_step_len):
        array_in[j][i] = data_ids_vectors[j+i]
    array_op[j] = data_ids_vectors[j+time_step_len]

train_len = int((len_dataset-time_step_len)*(1-test_train_split))
train_ft = array_in[0:train_len]
train_op = array_op[0:train_len]
test_ft = array_in[train_len:]
test_op = array_op[train_len:]
train_features = torch.from_numpy(train_ft)
test_features = torch.from_numpy(test_ft)
train_op = torch.from_numpy(train_op)
test_op = torch.from_numpy(test_op)

print(train_features.shape)
print(test_features.shape)
print(train_op.shape)
train = TensorDataset(train_features,train_op)
test = TensorDataset(test_features,test_op)

train_loader = DataLoader(train, batch_size = 512, shuffle = False)
test_loader = DataLoader(test, batch_size = 512, shuffle = False)

import torch 
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.RNN(input_size=len(set_), hidden_size=100, num_layers=2, nonlinearity="relu", bias=False, batch_first=True, dropout=0.5, bidirectional=False)
        self.fcn = torch.nn.Sequential(
            torch.nn.Linear(in_features=100, out_features=len(set_), bias=True)
        )
    
    def forward(self, x):
        output = self.rnn(x)
        return self.fcn(output[0])

device = torch.device('cuda:0')
model = Model().to(device)
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters())
error = torch.nn.CrossEntropyLoss()
loss = None
loss_list = []
max_accuracy = 0
for epoch in range(0,5000):
    loss_list = []
    for i, (features, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(features.float().to(device))
        labels = labels.to(device)
        loss = error(outputs[:,(time_step_len-1),:], labels.float())
        loss.backward()
        optimizer.step()
    final_accuracy = torch.tensor(np.zeros((len(test_loader),1)))
    for i, (features_, labels_) in enumerate(test_loader):
        outputs_ = model(features_.float().to(device))
        labels_ = labels_.to(device)
        output_indices = torch.argmax(outputs_[:,(time_step_len-1),:], dim=1)
        label_indices = torch.argmax(labels_[:,:], dim=1)
        final_accuracy[i] = torch.sum(output_indices==label_indices).to("cpu")
    print(final_accuracy.shape)
    accuracy = torch.sum(final_accuracy).item()
    print("Accuracy {} of epoch {}".format(accuracy, epoch))
    if accuracy > max_accuracy:
        print("Saving the model")
        torch.save(model.state_dict(), "./rnn_model_latest.pt".format(epoch))
        print("Model Saved")
        max_accuracy = accuracy





