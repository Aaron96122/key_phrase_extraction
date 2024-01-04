import re
import nltk
import os
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk import sent_tokenize
from stanfordcorenlp import StanfordCoreNLP
GRAMMAR = """  NP: {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""
stopword_dict = set(stopwords.words('english'))
StanfordCoreNLP_path = './stanford-corenlp-full-2018-10-05/'
en_model = StanfordCoreNLP(StanfordCoreNLP_path, quiet=True)

def extract_candidates(tokens_tagged):
    np_parser = nltk.RegexpParser(GRAMMAR)
    keyphrase_candidate = set()
    np_pos_tag_tokens = np_parser.parse(tokens_tagged)
    for token in np_pos_tag_tokens:
        if (isinstance(token, nltk.tree.Tree) and token._label == "NP"):
            np = ' '.join(word.lower() for word, tag in token.leaves())            
            keyphrase_candidate.add(np)       
    return keyphrase_candidate

def clean_text(text=""):
    pattern1 = re.compile(r'[<>[\]{}]')
    text = pattern1.sub(' ', text)
    text = text.replace("\t", " ")
    text = text.replace(' p ','\n')
    text = text.replace(' /p \n','\n')
    lines = text.splitlines()
    text_new=""
    for line in lines:
        if(line!='\n'):
            text_new+=line+'\n'
    return text_new

class InputTextObj:
    def __init__(self, en_model, file_path="./Test"):
        self.file_path = file_path
        self.en_model = en_model
    
    def get_data(self):
        self.data={}
        self.labels={}
        for dirname, dirnames, filenames in os.walk(self.file_path):
            for fname in filenames:
                left, right = fname.split('.')
                if (right == "abstr"):
                    infile = os.path.join(dirname, fname)
                    f=open(infile)
                    text=f.read()
                    text = text.replace("%", '')
                    text=clean_text(text)
                    self.data[left]=text
                if (right == "uncontr"):
                    infile = os.path.join(dirname, fname)
                    f=open(infile)
                    text=f.read()
                    text=text.replace("\n",' ')
                    text=clean_text(text)
                    text=text.lower().strip()
                    label=text.split("; ")
                    self.labels[left]=label
       
    def get_candidates(self):
        self.candidates = {}
        for key, text in self.data.items():
            tokens = []
            tokens_tagged = []
            tokens = self.en_model.word_tokenize(text)
            tokens_tagged = self.en_model.pos_tag(text)
            assert len(tokens) == len(tokens_tagged)
            for i, token in enumerate(tokens):
                if token.lower() in stopword_dict:
                    tokens_tagged[i] = (token, "IN")
            keyphrase_candidate = extract_candidates(tokens_tagged)
            self.candidates[key] = keyphrase_candidate
    
    def get_dataset(self):
        self.dataset = {}
        assert len(self.data) == len(self.labels) == len(self.candidates)
        for key, value in self.data.items():
            title = value.split('\n')[0]
            remainings = " ".join(value.split('\n')[1:]).replace('\n', '')
            sentences = sent_tokenize(remainings)
            sentences.append(title)
            self.dataset[key] = (sentences, self.candidates[key], self.labels[key])
        return self.dataset
    
def text_piece(text, en_model):
    if os.path.exists(text):
       with open(text, 'r') as f:
           content = f.readlines()
       data = " ".join([i.strip() for i in content])
    else:
       data = text
    tokens = []
    tokens_tagged = []
    tokens = en_model.word_tokenize(data)
    tokens_tagged = en_model.pos_tag(data)
    assert len(tokens) == len(tokens_tagged)
    for i, token in enumerate(tokens):
        if token.lower() in stopword_dict:
            tokens_tagged[i] = (token, "IN")
    keyphrase_candidate = extract_candidates(tokens_tagged)
    sentences = sent_tokenize(data)
    return sentences, keyphrase_candidate
        
        
