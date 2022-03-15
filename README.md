# TextSimilarity
A text cleaning and similarity ranking package.

## Purpose
The purpose of the textsimilarity package is to rank a text corpus based on similarity to a given target text. In this scenario the text should be words or short phrases. 

## Functionality 
This package provides functionality to:  
- clean text using the clean_text module  
- load a pre-trained natural language model using the text_models module  
- rank a corpus of text based on similarity to a given target text using the rankers module  

## Installation
You can install the textsimilarity package using pip.
```
pip install git+https://github.com/NalaniKai/TextSimilarity
```

Also, [install PyTorch](https://pytorch.org/) which on Windows is:
```
pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio===0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

## Usage
To use this package for comparing a corpus of text to a target.

First, import the required modules.
```
from textsimilarity import text_models, rankers
```

Second, create a model and ranker instance passing in your list of text.
```
#load a text model to use for generating text embeddings
bert_model = text_models.BertBaseModel()    

#specify corpus
comparison_corpus = ['relaxing vacation',
    'dancing and chocolate',
    'wedding party',
    'walking in the rain',
    'soccer game',
    'skate park']

#specify which model and corpus to use for comparison
cosine_sim_ranker = rankers.CosineSimilarityRanker(
                    bert_model, 
                    comparison_corpus
                    )
```

Third, rank the corpus based on another text.
```
#rank corpus based on target text
target = 'girls night out'
ranked_text = cosine_sim_ranker.rank_on_similarity(target)
```

Please refer to the [examples](https://github.com/NalaniKai/TextSimilarity/tree/main/examples) folder for more examples on how to use this package.