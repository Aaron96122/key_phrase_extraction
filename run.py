import argparse
import torch
from transformers import BertTokenizer, BertModel
from data import InputTextObj, extract_candidates, clean_text
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
from representation import MMRPhrase, Embeddings
import warnings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from data import text_piece
from result import stemmed, get_PRF
GRAMMAR = """  NP: {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""
stopword_dict = set(stopwords.words('english'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",
                        default='./Test',
                        type=str,
                        required=False,
                        help="The file path to the test dataset.")
    parser.add_argument("--StanfordCoreNLP_path",
                        default='./stanford-corenlp-full-2018-10-05/',
                        type=str,
                        required=True,
                        help="The path to the StanfordCoreNLP.")
    parser.add_argument("--model_path",
                        default='./bert-base-uncased',
                        type=str,
                        required=True,
                        help="The path to the pre-trained model or the name of the pre-trained model")
    parser.add_argument("--output",
                        default='./',
                        type=str,
                        required=False,
                        help="the path to the result file of the Test set")
    parser.add_argument("--beta",
                        default=1.0,
                        type=float,
                        required=False,
                        help="The tradeoff between similarity and diversity")
    parser.add_argument("--number",
                        default=5,
                        type=int,
                        required=False,
                        help="the number of key phrases extracted from the txt file")
    parser.add_argument("--text",
                        default=None,
                        type=str,
                        required=False,
                        help="the text you want to extract the key phrases for.")
    parser.add_argument("--sen_max_len",
                        default=30,
                        type=int,
                        required=False,
                        help="the max length set to embed sentences consisting the document.")
    parser.add_argument("--kpe_max_len",
                        default=10,
                        type=int,
                        required=False,
                        help="the max length set to embed the candidate key phrases.")
    args = parser.parse_args()
    return args

def main():
    args = parse_argument()
    StanfordCoreNLP_path = args.StanfordCoreNLP_path
    en_model = StanfordCoreNLP(StanfordCoreNLP_path, quiet=True)
    SEN_MAX_LEN = args.sen_max_len
    KPE_MAX_LEN = args.kpe_max_len
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    model = BertModel.from_pretrained(args.model_path, output_hidden_states = True)
    model = model.to(device)
    model.eval()
    if args.text:
        text = args.text
        sentences, candidates = text_piece(text, en_model=en_model)
        temp = Embeddings(model=model, tokenizer=tokenizer, sentences=sentences, candidates=list(candidates), sen_max_len=SEN_MAX_LEN, kpe_max_len=KPE_MAX_LEN)
        doc_embedd = temp.document_embeddings().cpu()
        can_embedd = temp.candidates_embeddings().cpu()
        candidates = temp.candidate()
        final,_, = MMRPhrase(doc_embedd, candidates, can_embedd, beta=args.beta, N=args.number, alias_threshold=0.8)
        print(f"The top {args.number} key phrases for you input text are:\n{final}")
    elif args.dataset_dir:
        data = InputTextObj(en_model=en_model, file_path=args.dataset_dir)
        data.get_data()
        data.get_candidates()
        dataset = data.get_dataset()
        final_candidates = {}
        final_labels = {}
        result = []
        for i in tqdm(dataset):
            output = {}
            sentences = dataset[i][0]
            candidates = dataset[i][1]
            label = dataset[i][2]
            temp = Embeddings(model=model, tokenizer=tokenizer, sentences=sentences, candidates=list(candidates), sen_max_len=SEN_MAX_LEN, kpe_max_len=KPE_MAX_LEN)
            doc_embedd = temp.document_embeddings().cpu()
            can_embedd = temp.candidates_embeddings().cpu()
            candidates = temp.candidate()
            final,_, = MMRPhrase(doc_embedd, candidates, can_embedd, beta=args.beta, N=args.number, alias_threshold=0.8)
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
        final = {'Precision':P,'Recall':R,'F1':F1}
        with open(f'{args.output}/PRF_{args.number}.json', 'w') as d:
             json.dump(final, d)

        with open(f'{args.output}/results_{args.number}.json', 'w') as f:
            for dictionary in result:
                json.dump(dictionary, f)
                f.write('\n')
    else:
        print("Please type in the text you want to extact the key phrases from or the file path to the Test set of Hulth datatset.")
        
if __name__ == "__main__":
    main()            
