import csv
import time

import torch
import networkx as nx
from node2vec import Node2Vec
import pickle
from scipy.spatial import cKDTree
import numpy as np
from sentence_transformers import SentenceTransformer, util

model_name = 'all-MiniLM-L6-v2'
# model_name = 'multi-qa-distilbert-cos-v1'
# model_name = 'all-distilroberta-v1'
# model_name = 'paraphrase-albert-small-v2'
model = SentenceTransformer(model_name)


kg_path = 'data/FB13/numeric_triples.tsv'
# G = nx.Graph()
#
# with open(kg_path, 'r', encoding='utf-8') as f:
#     data = csv.reader(f, delimiter='\t')
#     next(data)
#     for d in data:
#         head, tail = d[0], d[2]
#         G.add_edge(head, tail)
#
# with open('data/FB13/knowledge_graph_types.pkl', 'wb') as kg:
#     pickle.dump(G, kg)

path = 'data/FB13/entity_to_id.tsv'
embedding_ent = 'data/FB13/entity2vec.unif'
embedding_ent256 = 'data/FB13/entity2vec256.unif'
embedding_ent64 = 'data/FB13/entity2vec64.unif'
embedding_ent512 = 'data/FB13/entity2vec512.unif'
embedding_autosf_ent384 = 'data/FB13/entity2vec512_autosf.unif'
embedding_boxe_ent384 = 'data/FB13/entity2vec512_boxe.unif'
embedding_tucker_ent384 = 'data/FB13/entity2vec512_tucker.unif'

ent_text_path = 'data/FB13/entity2text.txt'
type_path = 'data/fb_ner_types.csv'
fb_question_rules = 'data/fb_question_rules.csv'
fb_question_ner = 'data/fb_question1.csv'
max_length = 384

ent2id, id2ent, ent2text, id2text, ent2type  = {}, {}, {}, {}, {}
types = []
question_rules = []
question_ner = []

with open(path, 'r', encoding='utf-8') as f:
    data = csv.reader(f, delimiter='\t')
    next(data)
    for d in data:
        id, ent = d[0], d[1]
        ent2id[ent] = int(id)
        id2ent[int(id)] = ent

with open(ent_text_path, 'r', encoding='utf-8') as f:
    data = f.readlines()
    for line in data:
        ent, text = line.strip('\n').split('\t')
        ent2text[ent] = text
        if ent in ent2id.keys():
            id2text[ent2id[ent]] = text

with open(type_path, 'r', encoding='utf-8') as f:
    data = csv.reader(f, delimiter='\t')
    for d in data:
        ent2type[d[0]] = d[2]
        if d[2] not in types:
            types.append(d[2])

with open(fb_question_rules, 'r', encoding='utf-8') as f:
    data = csv.reader(f, delimiter='\t')
    for d in data:
        question_rules.append([d[0], d[1], d[3]])

with open(fb_question_ner, 'r', encoding='utf-8') as f:
    data = csv.reader(f, delimiter='\t')
    for d in data:
        question_ner.append([d[0], d[2]])

# embeddings = np.loadtxt('data/FB13/entity2vec.unif', delimiter=' ')
embeddings = np.loadtxt(embedding_tucker_ent384, delimiter=' ')


def get_entity_from_rules(rule):
    rules = rule.split('^')
    entities, embed_entity = [], []
    for r in rules:
        try:
            h, pred, t = r.split('\t')
            # print(t)
            if str(h).replace(' ','_').lower() in ent2id.keys() and str(h).replace(' ','_').lower() not in entities:
                embed_entity.append(embeddings[ent2id[str(h).replace(' ','_').lower()]][0:max_length])
                entities.append(str(h).replace(' ','_').lower())
            if str(t).replace(' ','_').lower() in ent2id.keys() and str(t).replace(' ','_').lower() not in entities:
                embed_entity.append(embeddings[ent2id[str(t).replace(' ','_').lower()]][0:max_length])
                entities.append(str(t).replace(' ', '_').lower())
        except:
            continue
    return entities, embed_entity

def get_all_entities_from_rules(rule):
    rules = rule.split('^')
    ents = []
    for r in rules:
        try:
            h, pred, t = r.split('\t')
            if h not in ents:
                ents.append(h)
            if t not in ents:
                ents.append(t)
        except:
            continue
    return ents
# print(embeddings.shape)
# kdtree = cKDTree(embeddings)
# #
# with open('data/FB13/kdtree_fb_vector_512.pkl', 'wb') as kg:
#     pickle.dump(kdtree, kg)

with open('data/FB13/knowledge_graph_types.pkl', 'rb') as kg:
    G = pickle.load(kg)

with open('data/FB13/kdtree_fb_vector_512.pkl', 'rb') as kg:
    kdtree = pickle.load(kg)

def get_q_ner(question):
    for q in question_rules:
        if question == q[0]:
            return q
    return None

degrees = G.degree()
node2degree = {}
NodeDegree=[]
NodeName=[]
# Print the embeddings for each node
data=[]
NodeLabel=[]
points=[]
# Print the node degrees
for node, degree in degrees:
    # print(f"Node {node}: Degree {degree}")
    node2degree[int(node)] = degree

def check_entity_in_text(ents, stext):
    for e in ents:
        if e in stext:
            return True
    return False

ts = time.time()
i = 0
kgmodel = 'tucker'
llm = 'bert'
f = open(f'data/FB13/results384_{kgmodel}_{llm}.csv', 'w', encoding='utf-8', newline='')
w = csv.writer(f, delimiter='\t')

for q_n in question_ner[:1000]:
    i += 1
    s = time.time()
    q1 = q_n[0]
    ner = q_n[1]
    ques = get_q_ner(q1)
    if ques != None:
        question = model.encode(ques[0])[:max_length]
        entity, ent_in_rule = get_entity_from_rules(ques[2])

        all_ents = get_all_entities_from_rules(ques[2])
        query_point = embeddings[ent2id[ner]]
        num_neighbors = node2degree[ent2id[ner]]
        distances, indices = kdtree.query(query_point, k=num_neighbors)
        print(len(indices[1:]) - 1)
        # Print the nearest neighbors and their distances
        max = -1000
        answer_ent = ''
        j = 0
        f_distance = 0
        for distance, index in zip(distances[1:], indices[1:]):
            tem_ent = []
            if id2ent[index] not in types:
                # print(f"Nearest neighbor: {embeddings[index]}, Distance: {distance}, Node name: {id2ent[index]}")

                tem_ent.append(embeddings[index][0:max_length:1])

                # print(ent_in_rule)
                text = ''
                if id2ent[index] in ent2text.keys():
                    j += 1
                    text = ent2text[id2ent[index]]
                    if check_entity_in_text(all_ents, text) == True:
                        # with text
                        emb_text = model.encode(text)[:max_length]
                        tem_ent.append(emb_text)
                        arr = ent_in_rule + tem_ent

                        #without text
                        # arr = ent_in_rule + tem_ent

                        emb2 = torch.tensor(np.sum(arr, axis=0), dtype=torch.float32)
                        cos_sim = util.cos_sim(question, emb2)
                        rs = cos_sim.cpu().detach().numpy().item()
                        if rs > max:
                            max = rs
                            answer_ent = id2ent[index]

                        print(f'{ques[0]} -- {ques[1]} -- {id2ent[index]} -- {rs}')
                        if ques[1] == id2ent[index]:
                            answer_ent = id2ent[index]
                            f_distance = distance
                            break
                        else:
                            f_distance = distance

        e = time.time()
        print(len(indices[1:]))
        print(entity)

        if ques[1] == answer_ent:
            print(f'Question: {i}/{len(question_rules)} -- Answer: {ques[1]} -- {answer_ent} -- True -- Time: {e-s}  -- {j} -- {f_distance}')
            w.writerow([i, len(indices[1:]), ques[1], ent2type[ques[1]], answer_ent, ent2type[answer_ent] ,  True, e - s, j, f_distance])
        else:
            if answer_ent != '' and ques[1] in ent2id.keys():
                print(f'Question: {i}/{len(question_rules)} -- Answer: {ques[1]} -- {answer_ent} -- False -- Time: {e-s}')
                w.writerow([i,len(indices[1:]), ques[1], ent2type[ques[1]], answer_ent, ent2type[answer_ent], False, e - s, j , f_distance])
f.close()
te = time.time()
print(f'Total time {te-ts}')


