import torch
from transformers import BertTokenizer, BertModel
from data import InputTextObj, extract_candidates, clean_text, text_piece
from result import stemmed, get_PRF
import re
import nltk
import os
from scipy.spatial.distance import cosine
import csv
import json
from tqdm import tqdm
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk import sent_tokenize
from stanfordcorenlp import StanfordCoreNLP
import warnings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
GRAMMAR = """  NP: {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""
stopword_dict = set(stopwords.words('english'))
StanfordCoreNLP_path = './stanford-corenlp-full-2018-10-05/'
en_model = StanfordCoreNLP(StanfordCoreNLP_path, quiet=True)
SEN_MAX_LEN = 30
KPE_MAX_LEN = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def max_normalization(array):
    return 1/np.max(array) * array.squeeze(axis=1)


def _MMR(doc_embedd, candidates, can_embedd, beta, N, alias_threshold):
    candidates = np.array(candidates)
    N = min(N, len(candidates)) 
    doc_sim = cosine_similarity(can_embedd, doc_embedd.reshape(1, -1))
    doc_sim_norm = doc_sim/np.max(doc_sim)
    doc_sim_norm = 0.5 + (doc_sim_norm - np.average(doc_sim_norm)) / np.std(doc_sim_norm)
    sim_between = cosine_similarity(can_embedd)
    np.fill_diagonal(sim_between, np.NaN)
    sim_between_norm = sim_between/np.nanmax(sim_between, axis=0)
    sim_between_norm = \
        0.5 + (sim_between_norm - np.nanmean(sim_between_norm, axis=0)) / np.nanstd(sim_between_norm, axis=0)
    selected_candidates = []
    unselected_candidates = [c for c in range(len(candidates))]
    j = np.argmax(doc_sim)
    selected_candidates.append(j)
    unselected_candidates.remove(j)
    for _ in range(N - 1):
        selec_array = np.array(selected_candidates)
        unselec_array = np.array(unselected_candidates)
        distance_to_doc = doc_sim_norm[unselec_array, :]
        dist_between = sim_between_norm[unselec_array][:, selec_array]
        if dist_between.ndim == 1:
            dist_between = dist_between[:, np.newaxis]
        j = np.argmax(beta * distance_to_doc - (1 - beta) * np.max(dist_between, axis=1).reshape(-1, 1))
        item_idx = unselected_candidates[j]
        selected_candidates.append(item_idx)
        unselected_candidates.remove(item_idx)
    relevance_list = max_normalization(doc_sim[selected_candidates]).tolist()
    return candidates[np.array(selected_candidates)].tolist(), relevance_list

def MMRPhrase(doc_embedd, candidates, can_embedd, beta=0.65, N=10, alias_threshold=0.8):
    if len(candidates) == 0:
        warnings.warn('No keyphrase extracted for this document')
        return None, None, None

    return _MMR(doc_embedd, candidates, can_embedd, beta, N, alias_threshold)

class Embeddings:
    def __init__(self, model, tokenizer, sentences, candidates, sen_max_len=SEN_MAX_LEN, kpe_max_len=KPE_MAX_LEN, layer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.sentences = sentences 
        self.candidates = candidates
        self.sen_max_len = sen_max_len
        self.kpe_max_len = kpe_max_len
        self.layer = layer
    
    def document_embeddings(self):
        input_ids = []
        attention_masks = []
        for sentence in self.sentences:
            encoded_dict = self.tokenizer.encode_plus(
                                sentence,                      
                                add_special_tokens = True, 
                                truncation = True,
                                max_length = self.sen_max_len,
                                padding = 'max_length',
                                return_attention_mask = True,   
                                return_tensors = 'pt',     
                        )  
            input_ids.append(encoded_dict['input_ids'].to(device))
            attention_masks.append(encoded_dict['attention_mask'].to(device))
        
        hidden_states=[]
        with torch.no_grad():
            if self.layer:
                for i in range(len(input_ids)):
                    outputs = self.model(input_ids[i],attention_mask=attention_masks[i])
                    hidden_states.append(outputs.hidden_states[self.layer])
            else:
                for i in range(len(input_ids)):
                    outputs = self.model(input_ids[i],attention_mask=attention_masks[i])
                    hidden_states.append(outputs.last_hidden_state)

        sentence_embeddings=[]
        for i in range(len(hidden_states)):
            token_vecs = hidden_states[i][0]
            sentence_embedding = torch.mean(token_vecs, dim=0)
            sentence_embeddings.append(sentence_embedding.to(device))

        sentence_embeddings = torch.stack(sentence_embeddings,dim=0)
        self.document_embedding=torch.mean(sentence_embeddings, dim=0)
        return self.document_embedding
    
    def candidates_embeddings(self):
        input_ids = []
        attention_masks = []
        for candidate in self.candidates:
            encoded_dict = self.tokenizer.encode_plus(
                                candidate,                      
                                add_special_tokens = True, 
                                truncation = True,
                                max_length = self.kpe_max_len,
                                padding = 'max_length',
                                return_attention_mask = True,   
                                return_tensors = 'pt',    
                        )   
            input_ids.append(encoded_dict['input_ids'].to(device))
            attention_masks.append(encoded_dict['attention_mask'].to(device))
        
        hidden_states=[]
        with torch.no_grad():
            if self.layer:
                for i in range(len(input_ids)):
                    outputs = self.model(input_ids[i],attention_mask=attention_masks[i])
                    hidden_states.append(outputs.hidden_states[self.layer])
            else:
                for i in range(len(input_ids)):
                    outputs = self.model(input_ids[i],attention_mask=attention_masks[i])
                    hidden_states.append(outputs.last_hidden_state)
        
        self.can_embeddings=[]
        for i in range(len(hidden_states)):
            token_vecs = hidden_states[i][0]
            can_embedding = torch.mean(token_vecs, dim=0)
            self.can_embeddings.append(can_embedding.to(device))
        self.can_embeddings = torch.stack(self.can_embeddings,dim=0)
        return self.can_embeddings
    
    def candidate(self):
        return self.candidates



if __name__ == "__main__": 
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
    model.eval()
    data = InputTextObj(en_model=en_model, file_path="./Test")
    data.get_data()
    data.get_candidates()
    dataset = data.get_dataset()
    dataset = data.get_dataset()
    final_candidates = {}
    final_labels = {}
    result = []
    for i in tqdm(dataset):
        output = {}
        sentences = dataset[i][0]
        candidates = dataset[i][1]
        label = dataset[i][2]
        temp = Embeddings(model=model, tokenizer=tokenizer, sentences=sentences, candidates=list(candidates), layer=6)
        doc_embedd = temp.document_embeddings()
        can_embedd = temp.candidates_embeddings()
        candidates = temp.candidate()
        final,_, = MMRPhrase(doc_embedd, candidates, can_embedd, beta=1, N=15, alias_threshold=0.8)
        final_candidates[i] = final
        final_labels[i] = label
        output['name'] = i
        output['candidates'] = final
        output['labels'] = label
        result.append(output)
    c = []
    l = []
    for k,v in final_candidates.items():
        c.append(list(set(stemmed(v))))
        l.append(list(set(stemmed(final_labels[k]))))
    assert len(c) == len(l)
    P, R, F1 = get_PRF(candidates=c, labels=l)
    print(f"Precison:{P},\n Recall:{R},\n F1 score:{F1}")



