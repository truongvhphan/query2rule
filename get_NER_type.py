import spacy
import csv

nlp = spacy.load('en_core_web_sm')

path = 'data/YAGO13/entities.txt'
write = open('data/yago_ner_types.csv', 'w', encoding='utf-8', newline='')
w = csv.writer(write, delimiter='\t')
with open(path, 'r', encoding='utf-8') as f:
    data = f.readlines()

    for d in data:
        ## FB
        # id, ner = d.strip('\n').split('\t')
        # doc = nlp(ner.replace('_', ' '))
        # for ent in doc.ents:
        #     print(f'{ent} -- {ent.label_}')
        #     w.writerow([str(ent).replace(' ','_'),'1', ent.label_])

        ## YAGO
        ner = d.strip('\n')
        doc = nlp(ner.replace('_', ' '))
        for ent in doc.ents:
            print(f'{ent} -- {ent.label_}')
            w.writerow([str(ent).replace(' ','_'),'1', ent.label_])