# BERT NER

Use [Mongolian pre-trained BERT](https://github.com/tugstugi/mongolian-bert) for finetuning [NER](https://en.wikipedia.org/wiki/Named-entity_recognition) task on [Mongolian NER dataset](https://github.com/tugstugi/mongolian-nlp/blob/master/datasets/NER_v1.0.json.gz)


# Requirements
First download pre-trained cased BERT-Base model from [here](https://drive.google.com/file/d/11Adpo6DorPgpE8z1lL6rvZAMHLEfnJwv)

-  `python3`
- `pip3 install -r requirements.txt`

Rename following file for compatibility:
```
mv cased_bert_base_pytorch/bert_config.json cased_bert_base_pytorch/config.json
```

# Run

`python run_ner.py --data_dir=data/ --bert_model=cased_bert_base_pytorch --task_name=ner --output_dir=out --max_seq_length=50 --do_train --num_train_epochs 5 --do_eval --warmup_proportion=0.1`


# Result
## BERT
Following evaluations done by using [seqeval](https://github.com/chakki-works/seqeval) sequence labeling toolkit.
### Validation Data
```
             precision    recall  f1-score   support

        LOC     0.8321    0.8630    0.8473       511
       MISC     0.6790    0.7746    0.7237       213
        PER     0.8053    0.8828    0.8423       239
        ORG     0.7952    0.8399    0.8169       356

avg / total     0.7926    0.8461    0.8182      1319
```
### Test Data
```
             precision    recall  f1-score   support

        ORG     0.7761    0.8266    0.8005       369
       MISC     0.6851    0.7444    0.7135       266
        LOC     0.7894    0.8537    0.8203       540
        PER     0.8287    0.8681    0.8479       273

avg / total     0.7742    0.8294    0.8008      1448
```
## [2016 NER SOTA](https://www.aclweb.org/anthology/P16-1101) 
Used NCRF++ toolkit(refer to NCRFpp directory) with a combination of `Char CNN + Word LSTM + CRF` 
### Validation Data
```
             precision    recall  f1-score   support

        ORG       0.72      0.58      0.64       356
       MISC       0.53      0.46      0.49       213
        LOC       0.73      0.72      0.73       511
        PER       0.61      0.62      0.61       239

avg / total       0.67      0.62      0.65      1319
```
### Test Data
```
             precision    recall  f1-score   support

        PER       0.60      0.61      0.60       273
        LOC       0.73      0.71      0.72       540
        ORG       0.70      0.53      0.60       369
       MISC       0.59      0.49      0.54       266

avg / total       0.67      0.61      0.64      1448
```
## [sklearn-crfsuite](https://sklearn-crfsuite.readthedocs.io/en/latest/)
### Validation Data
```
           precision    recall  f1-score   support

      LOC       0.53      0.45      0.49       511
      ORG       0.53      0.39      0.45       356
      PER       0.53      0.39      0.45       239
     MISC       0.39      0.21      0.27       213

micro avg       0.51      0.38      0.44      1319
macro avg       0.51      0.38      0.43      1319
```
### Test Data
```
           precision    recall  f1-score   support

      LOC       0.54      0.46      0.50       540
      ORG       0.54      0.40      0.46       369
     MISC       0.41      0.25      0.31       266
      PER       0.55      0.43      0.48       273

micro avg       0.52      0.40      0.45      1448
macro avg       0.52      0.40      0.45      1448
```

## Pre-trained NER model - download from [here](https://drive.google.com/open?id=1pCvITS3ciu-h10toW868rOviQbrTjBFn)
Then unzip inside in this repo. 

# Run prediction inside python module

```python
from bert import Ner

model = Ner("out/")

output = model.predict("АТГ-аас сар бүр хийдэг хэвлэлийн хурлаа өнөөдөр хийлээ. Энэ үеэр Мөрдөн шалгах хэлтсийн дарга Д.Батбаяр сэтгүүлчдийн асуултад хариулсан юм.")

print(output)
# {
# 	'АТГ-аас': {'tag': 'B-ORG', 'confidence': 0.999990701675415}, 
# 	'сар': {'tag': 'O', 'confidence': 0.991750180721283}, 
# 	'бүр': {'tag': 'O', 'confidence': 0.9999933242797852}, 
# 	'хийдэг': {'tag': 'O', 'confidence': 0.9999896287918091}, 
# 	'хэвлэлийн': {'tag': 'O', 'confidence': 0.9999939203262329}, 
# 	'хурлаа': {'tag': 'O', 'confidence': 0.9999923706054688}, 
# 	'өнөөдөр': {'tag': 'O', 'confidence': 0.9999933242797852}, 
# 	'хийлээ': {'tag': 'O', 'confidence': 0.9999940395355225}, 
# 	'.': {'tag': 'O', 'confidence': 0.9999922513961792}, 
# 	'Энэ': {'tag': 'O', 'confidence': 0.9999942779541016}, 
# 	'үеэр': {'tag': 'O', 'confidence': 0.9999926090240479}, 
# 	'Мөрдөн': {'tag': 'B-ORG', 'confidence': 0.9999772310256958}, 
# 	'шалгах': {'tag': 'I-ORG', 'confidence': 0.9999890327453613}, 
# 	'хэлтсийн': {'tag': 'I-ORG', 'confidence': 0.8935487270355225}, 
# 	'дарга': {'tag': 'O', 'confidence': 0.9999908208847046}, 
# 	'Д.Батбаяр': {'tag': 'B-PER', 'confidence': 0.9998291730880737}, 
# 	'сэтгүүлчдийн': {'tag': 'O', 'confidence': 0.9998449087142944}, 
# 	'асуултад': {'tag': 'O', 'confidence': 0.999796450138092}, 
# 	'хариулсан': {'tag': 'O', 'confidence': 0.9999463558197021}, 
# 	'юм': {'tag': 'O', 'confidence': 0.9513341784477234}
# }
```
# Run web app to predict

Run `python app.py` - runs web server on http://localhost:5000/ 

![Flak webapp](images/image.png)

# Train Valid Test split
Refer to `notebook/CoNLL conversion.ipynb` file.

# ToDo
- [ ] [Add 2017 SOTA](https://arxiv.org/pdf/1709.04109.pdf)
- [ ] Add either ELMo/ULMFIT
- [ ] Change webapp inferface to [AllenNLP demo](https://demo.allennlp.org/named-entity-recognition)