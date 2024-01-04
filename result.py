from nltk import PorterStemmer
import pandas as pd
import csv
import json
poter = PorterStemmer()
def stemmed(lst):
    label_s = []
    for l in lst:
        tokens = l.split()
        label_s.append(" ".join(poter.stem(t) for t in tokens))
    return label_s

def get_PRF(candidates, labels):
    sum_e = []
    sum_c = [token for candidate in candidates for token in candidate]
    sum_l = [token for label in labels for token in label]
    for i in range(len(candidates)):
        for token in candidates[i]:
            if token in labels[i]:
                sum_e.append(token)
    P = len(sum_e) / len(sum_c)
    R = len(sum_e) / len(sum_l)
    if (P + R == 0.0):
        F1 = 0
    else:
        F1 = 2 * P * R / (P + R)
    return P, R, F1


        

if __name__ == "__main__":
    candidates_unstemmed = []
    labels = []
    with open('./result_15_layer_1.csv', 'r') as file:
        next(file)
        reader = csv.reader(file)
        deserialized_list_of_dicts = []
        for row in reader:
            candidate = json.loads(row[1])
            candidates_unstemmed.append(candidate)
            label = set(stemmed(json.loads(row[2])))
            labels.append(list(label))
    assert len(candidates_unstemmed) == len(labels)
    whole = []
    for length in range(5,16):
        test_dict = {}
        candidate_stemmed = [list(set(stemmed(i[:length]))) for i in candidates_unstemmed]
        P, R, F1 = get_PRF(candidates=candidate_stemmed, labels=labels)
        test_dict = {'number':length, 'precision':P, 'recall':R, 'F1':F1}
        whole.append(test_dict)
    with open("./result_bert_layer_1.json","w") as f:
        for line in whole:
            json.dump(line,f)
            f.write('\n')

    
    
    

