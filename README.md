# Key Phrase Extraction project

This project is the realization of the Key Phrase Extraction using the [Inspect](https://github.com/snkim/AutomaticKeyphraseExtraction/blob/master/Hulth2003.tar.gz).

## Environment

```
StanfordCoreNLP 3.9.1.1
Python 3.8
torch 1.9.1
nltk 3.7
transformers 4.23
```

## Usage

First, download the full Stanford CoreNLP Tagger version 3.8.0 (http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip)


Second, we use run.sh script to replicate the experiment on the test set of the Inspect dataset.

```
sh run.sh
```

```
arguments:
  --dataset_dir          The path to the test set of Inspect. 
  --StanfordCoreNLP_path The path to the downloaded Stanford CoreNLP Tagger.
  --model_path           The path to the pre-trained model or the model_name of a bert model (e.g.  bert-base-uncased)
                        
  --output              The directory to store the final result.
  --beta                The coefficient of beta for the diversity. 
  --number              The number of key phrases extracted from each document.
```

Or, we can run run_text.sh script to experiment with a single text file or a string of text.

```
sh run.sh
```

```
arguments:
  --text                 The string of text or the path to the file of .txt.
  --StanfordCoreNLP_path The path to the downloaded Stanford CoreNLP Tagger.
  --model_path           The path to the pre-trained model or the model_name of a bert model (e.g.  bert-base-uncased)
                        
  --beta                The coefficient of beta for the diversity. 
  --number              The number of key phrases extracted from each document.
```





