import csv
import json
import spacy

nlp = spacy.load('en_core_web_sm')
path = 'FreebaseQA-dev.json'
eval_path = 'FreebaseQA-eval.json'
partial_path = 'FreebaseQA-partial.json'
train_path = 'FreebaseQA-train.json'
fb13_path = 'FB13/entities.txt'
yago13_path = 'YAGO13/entities.txt'
entites = []
with open(yago13_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for i in lines:
        entites.append(i.strip('\n').lower())


with open(partial_path) as f:
    data = json.load(f)

answer_fb_entity = []
# f = open('fb_question.csv','w', encoding='utf-8', newline='')
# w = csv.writer(f, delimiter='\t')
# for i in data['Questions']:
#     answer_entity = i['Parses'][0]['Answers'][0]['AnswersName'][0].replace(' ', '_')
#     if answer_entity in entites:
#         answer_fb_entity.append([i['RawQuestion'], answer_entity])
#         doc = nlp(i['Parses'][0]['Answers'][0]['AnswersName'][0])
#         label = ''
#         for ent in doc.ents:
#             label = ent.label_
#             break
#         if label != '':
#             w.writerow([i['RawQuestion'], answer_entity, label])
#         print(f"{i['RawQuestion']} ------ {answer_entity} -------- {label}")
#
# print(len(answer_fb_entity))

f = open('yago_question.csv', 'a', encoding='utf-8', newline='')
w = csv.writer(f, delimiter='\t')
for i in data['Questions']:
    answer_entity = i['Parses'][0]['Answers'][0]['AnswersName'][0].replace(' ', '_')
    answer_fb_entity.append([i['RawQuestion'], answer_entity])
    doc = nlp(i['Parses'][0]['Answers'][0]['AnswersName'][0])
    label = ''
    for ent in doc.ents:
        label = ent.label_
        break
    if label != '':
        w.writerow([i['RawQuestion'], answer_entity, label])
    print(f"{i['RawQuestion']} ------ {answer_entity} -------- {label}")

print(len(answer_fb_entity))