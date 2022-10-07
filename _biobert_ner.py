#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import re
import csv
import itertools

import nltk
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertConfig
from transformers import BertForTokenClassification, AdamW
from transformers import get_linear_schedule_with_warmup

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


# In[11]:


import pytorch_pretrained_bert
import wget


# In[12]:


# Get GPU device name
device_name = tf.test.gpu_device_name()

if device_name == '/device:GPU:0':
    print('Found GPU at: {}'.format(device_name))
else:
    raise SystemError('GPU device not found')


# In[13]:


# tell Pytorch to use the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('There are %d GPU(s) available.' % torch.cuda.device_count())
print('We will use the GPU:', torch.cuda.get_device_name(0))


# In[14]:


get_ipython().system('transformers-cli convert --model_type bert --tf_checkpoint biobert_v1.1_pubmed/model.ckpt-1000000 --config biobert_v1.1_pubmed/bert_config.json --pytorch_dump_output biobert_v1.1_pubmed/pytorch_model.bin')


# In[15]:


get_ipython().system('dir biobert_v1.1_pubmed/')
get_ipython().system('move biobert_v1.1_pubmed/bert_config.json biobert_v1.1_pubmed/config.json')
get_ipython().system('dir biobert_v1.1_pubmed/')


# In[16]:


get_ipython().system('dir')


# In[17]:


MAX_LEN = 75
BATCH_SIZE = 32
tokenizer = BertTokenizer(vocab_file='biobert_v1.1_pubmed/vocab.txt', do_lower_case=False)


# In[18]:


data = pd.read_csv('./bionlp_tags.csv')
tag_values = data['tags'].values
vocab_len = len(tag_values)
print('Entity Types:',vocab_len)


# In[19]:


df_tags = pd.DataFrame({'tags':tag_values})
df_tags.to_csv('bionlp_tags.csv',index=False)
df = pd.read_csv('bionlp_tags.csv')
print('Tag Preview:\n', df)


# In[20]:


class SentenceFetch(object):
  
    def __init__(self, data):
        self.data = data
        self.sentences = []
        self.tags = []
        self.sent = []
        self.tag = []

        # make tsv file readable
        with open(self.data) as tsv_f:
            reader = csv.reader(tsv_f, delimiter='\t')
            for row in reader:
                if len(row) == 0:
                    if len(self.sent) != len(self.tag):
                        break
                    self.sentences.append(self.sent)
                    self.tags.append(self.tag)
                    self.sent = []
                    self.tag = []
                else:
                    self.sent.append(row[0])
                    self.tag.append(row[1])   

    def getSentences(self):
        return self.sentences

    def getTags(self):
        return self.tags


# In[21]:


corpora = './Data/bionlp_corpora'
sentences = []
tags = []
for subdir, dirs, files in os.walk(corpora):
    for file in files:
        if file == 'train.tsv':
            path = os.path.join(subdir, file)
            sent = SentenceFetch(path).getSentences()
            tag = SentenceFetch(path).getTags()
            sentences.extend(sent)
            tags.extend(tag)
            
sentences = sentences[0:20000]
tags = tags[0:20000]


# In[22]:


sentences


# In[23]:


print('Sentence Preview:\n',sentences[0])


# In[24]:


def tok_with_labels(sent, text_labels):
    '''tokenize and keep labels intact'''
    tok_sent = []
    labels = []
    for word, label in zip(sent, text_labels):
        tok_word = tokenizer.tokenize(word)
        n_subwords = len(tok_word)

        tok_sent.extend(tok_word)
        labels.extend([label] * n_subwords)
    return tok_sent, labels

tok_texts_and_labels = [tok_with_labels(sent, labs) for sent, labs in zip(sentences, tags)]


# In[25]:


tok_texts = [tok_label_pair[0] for tok_label_pair in tok_texts_and_labels]
labels = [tok_label_pair[1] for tok_label_pair in tok_texts_and_labels]


# In[26]:


len(tok_texts)


# In[27]:


input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tok_texts],
                          maxlen=MAX_LEN, dtype="long", value=0.0,
                          truncating="post", padding="post")


# In[28]:


print('WordPiece Tokenizer Preview:\n', tok_texts[0])


# In[29]:


tag_values = list(set(itertools.chain.from_iterable(tags)))
tag_values.append("PAD")

tag2idx = {t: i for i,t in enumerate(tag_values)}


# In[30]:


tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                     maxlen=MAX_LEN, value=tag2idx["PAD"], padding="post",
                     dtype="long", truncating="post")


# In[31]:


# attention masks make explicit reference to which tokens are actual words vs padded words
attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]


# In[32]:


tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                            random_state=2018, test_size=0.1)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=0.1)

tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)


# In[33]:


train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE)


# In[35]:


config = BertConfig.from_json_file('./biobert_v1.1_pubmed/bert_config.json')
tmp_d = torch.load('biobert_v1.1_pubmed/pytorch_model.bin', map_location=device)
state_dict = OrderedDict()

for i in list(tmp_d.keys())[:199]:
    x = i
    if i.find('bert') > -1:
        x = '.'.join(i.split('.')[1:])
    state_dict[x] = tmp_d[i]


# In[36]:


class BioBertNER(nn.Module):

    def __init__(self, vocab_len, config, state_dict):
        super().__init__()
        self.bert = BertModel(config)
        self.bert.load_state_dict(state_dict, strict=False)
        self.dropout = nn.Dropout(p=0.3)
        self.output = nn.Linear(self.bert.config.hidden_size, vocab_len)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        encoded_layer, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        encl = encoded_layer[-1]
        out = self.dropout(encl)
        out = self.output(out)
        return out, out.argmax(-1)


# In[37]:


model = BioBertNER(vocab_len,config,state_dict)
model.to(device)


# In[41]:


param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=3e-5,
    eps=1e-8
)
epochs = 10
max_grad_norm = 1.0

total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss().to(device)


# In[42]:


def train_epoch(model,data_loader,loss_fn,optimizer,device,scheduler):
    model = model.train()
    losses = []
    correct_predictions = 0
    for step,batch in enumerate(data_loader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        outputs,y_hat = model(b_input_ids,b_input_mask)
        
        _,preds = torch.max(outputs,dim=2)
        outputs = outputs.view(-1,outputs.shape[-1])
        b_labels_shaped = b_labels.view(-1)
        loss = loss_fn(outputs,b_labels_shaped)
        correct_predictions += torch.sum(preds == b_labels)
        losses.append(loss.item())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
    return correct_predictions.double()/len(data_loader) , np.mean(losses)


# In[43]:


def model_eval(model,data_loader,loss_fn,device):
    model = model.eval()
    
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
        
            outputs,y_hat = model(b_input_ids,b_input_mask)
        
            _,preds = torch.max(outputs,dim=2)
            outputs = outputs.view(-1,outputs.shape[-1])
            b_labels_shaped = b_labels.view(-1)
            loss = loss_fn(outputs,b_labels_shaped)
            correct_predictions += torch.sum(preds == b_labels)
            losses.append(loss.item())
        
    
    return correct_predictions.double()/len(data_loader) , np.mean(losses)


# In[44]:


get_ipython().run_cell_magic('time', '', "history = defaultdict(list)\nbest_accuracy = 0\nnormalizer = BATCH_SIZE*MAX_LEN\nloss_values = []\n\nfor epoch in range(epochs):\n    \n    total_loss = 0\n    print(f'======== Epoch {epoch+1}/{epochs} ========')\n    train_acc,train_loss = train_epoch(model,train_dataloader,loss_fn,optimizer,device,scheduler)\n    train_acc = train_acc/normalizer\n    print(f'Train Loss: {train_loss} Train Accuracy: {train_acc}')\n    total_loss += train_loss.item()\n    \n    avg_train_loss = total_loss / len(train_dataloader)  \n    loss_values.append(avg_train_loss)\n    \n    val_acc,val_loss = model_eval(model,valid_dataloader,loss_fn,device)\n    val_acc = val_acc/normalizer\n    print(f'Val Loss: {val_loss} Val Accuracy: {val_acc}')\n    \n    history['train_loss'].append(train_loss)\n    history['train_acc'].append(train_acc)\n    \n    history['val_loss'].append(val_loss)\n    history['val_acc'].append(val_acc)")


# In[45]:


sns.set(style='darkgrid')

sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)

# learning curve
plt.plot(loss_values, 'b-o')

plt.title("Training loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.show()


# Test

# In[46]:


text = """In addition to their essential catalytic role in protein biosynthesis, aminoacyl-tRNA synthetases participate in numerous other functions, including regulation of gene expression and amino acid biosynthesis via transamidation pathways. Herein, we describe a class of aminoacyl-tRNA synthetase-like (HisZ) proteins based on the catalytic core of the contemporary class II histidyl-tRNA synthetase whose members lack aminoacylation activity but are instead essential components of the first enzyme in histidine biosynthesis ATP phosphoribosyltransferase (HisG). Prediction of the function of HisZ in Lactococcus lactis was assisted by comparative genomics, a technique that revealed a link between the presence or the absence of HisZ and a systematic variation in the length of the HisG polypeptide. HisZ is required for histidine prototrophy, and three other lines of evidence support the direct involvement of HisZ in the transferase function. (i) Genetic experiments demonstrate that complementation of an in-frame deletion of HisG from Escherichia coli (which does not possess HisZ) requires both HisG and HisZ from L. lactis. (ii) Coelution of HisG and HisZ during affinity chromatography provides evidence of direct physical interaction. (iii) Both HisG and HisZ are required for catalysis of the ATP phosphoribosyltransferase reaction. This observation of a common protein domain linking amino acid biosynthesis and protein synthesis implies an early connection between the biosynthesis of amino acids and proteins."""


# In[48]:


text


# In[49]:


nltk.download('punkt')


# In[50]:


sent_text = nltk.sent_tokenize(text)


# In[52]:


tokenized_text = []
for sentence in sent_text:
    tokenized_text.append(nltk.word_tokenize(sentence))


# In[53]:


def tokenize_and_preserve(sentence):
    tokenized_sentence = []
    
    for word in sentence:
        tokenized_word = tokenizer.tokenize(word)   
        tokenized_sentence.extend(tokenized_word)

    return tokenized_sentence


# In[54]:


tok_texts = [
    tokenize_and_preserve(sent) for sent in tokenized_text
]


# In[55]:


input_ids = [tokenizer.convert_tokens_to_ids(txt) for txt in tok_texts]
input_attentions = [[1]*len(in_id) for in_id in input_ids]


# In[56]:


tokens = tokenizer.convert_ids_to_tokens(input_ids[1])
new_tokens, new_labels = [], []
for token in tokens:
    if token.startswith("##"):
        new_tokens[-1] = new_tokens[-1] + token[2:]
    else:
        
        new_tokens.append(token)


# In[57]:


actual_sentences = []
pred_labels = []
for x,y in zip(input_ids,input_attentions):
    x = torch.tensor(x).cuda()
    y = torch.tensor(y).cuda()
    x = x.view(-1,x.size()[-1])
    y = y.view(-1,y.size()[-1])
    with torch.no_grad():
        _,y_hat = model(x,y)
    label_indices = y_hat.to('cpu').numpy()
    
    tokens = tokenizer.convert_ids_to_tokens(x.to('cpu').numpy()[0])
    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, label_indices[0]):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(tag_values[label_idx])
            new_tokens.append(token)
    actual_sentences.append(new_tokens)
    pred_labels.append(new_labels)


# In[58]:


for token, label in zip(actual_sentences, pred_labels):
    for t,l in zip(token,label):
        print("{}\t{}".format(t, l))


# In[60]:


model_save = 'BIONER_classifier.pt'
path = F"models/{model_save}" 
torch.save(model.state_dict(), path)


# In[ ]:




