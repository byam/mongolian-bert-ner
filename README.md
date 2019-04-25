# BERT NER

Use [Mongolian pre-trained BERT](https://github.com/tugstugi/mongolian-bert) for finetuning [NER](https://en.wikipedia.org/wiki/Named-entity_recognition) task on [Mongolian NER dataset](https://github.com/tugstugi/mongolian-nlp/blob/master/datasets/NER_v1.0.json.gz)


# Requirements
First download pre-trained cased BERT-Base model from [here](https://drive.google.com/file/d/11Adpo6DorPgpE8z1lL6rvZAMHLEfnJwv)

-  `python3`
- `pip3 install -r requirements.txt`

# Run

`python run_ner.py --data_dir=data/ --bert_model=bert-base-cased --task_name=ner --output_dir=out --max_seq_length=50 --do_train --num_train_epochs 5 --do_eval --do_test --warmup_proportion=0.4`


# Result

### Validation Data
```
             precision    recall  f1-score   support

        LOC     0.8710    0.9310    0.9000       232
       MISC     0.7838    0.7945    0.7891        73
        PER     0.9130    0.9545    0.9333        22
        ORG     0.8043    0.7872    0.7957        94

avg / total     0.8432    0.8765    0.8592       421
```
### Test Data
```
             precision    recall  f1-score   support

        ORG     0.7411    0.8300    0.7830       100
        LOC     0.8340    0.8852    0.8588       244
        PER     0.8182    0.8438    0.8308        32
       MISC     0.6591    0.7632    0.7073        76

avg / total     0.7829    0.8496    0.8146       452
```

## Pretrained model download from [here](https://drive.google.com/open?id=1pCvITS3ciu-h10toW868rOviQbrTjBFn)
Then unzip inside in this repo. 

# Prediction on given text

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

# Train Valid Test split
Refer to `notebook/CoNLL conversion.ipynb` file.
