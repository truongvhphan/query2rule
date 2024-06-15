import csv

import networkx as nx
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import pipeline
import math
import torch
import wikipediaapi
from sentence_transformers import SentenceTransformer, util

s_model = SentenceTransformer('all-MiniLM-L6-v2')
wiki_wiki = wikipediaapi.Wikipedia('query2rule', 'en')
tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

def extract_relations_from_model_output(text):
    relations = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
    for token in text_replaced.split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        relations.append({
            'head': subject.strip(),
            'type': relation.strip(),
            'tail': object_.strip()
        })
    return relations

class KB():
    def __init__(self):
        self.relations = []

    def are_relations_equal(self, r1, r2):
        return all(r1[attr] == r2[attr] for attr in ["head", "type", "tail"])

    def exists_relation(self, r1):
        return any(self.are_relations_equal(r1, r2) for r2 in self.relations)

    def add_relation(self, r):
        if not self.exists_relation(r):
            self.relations.append(r)

    def print(self):
        print("Relations:")
        for r in self.relations:
            print(f"  {r}")

def from_small_text_to_kb(text, verbose=False):
    kb = KB()

    # Tokenizer text
    model_inputs = tokenizer(text, max_length=512, padding=True, truncation=True,
                            return_tensors='pt')
    if verbose:
        print(f"Num tokens: {len(model_inputs['input_ids'][0])}")

    # Generate
    gen_kwargs = {
        "max_length": 216,
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": 3
    }
    generated_tokens = model.generate(
        **model_inputs,
        **gen_kwargs,
    )
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

    # create kb
    for sentence_pred in decoded_preds:
        relations = extract_relations_from_model_output(sentence_pred)
        for r in relations:
            kb.add_relation(r)

    return kb


fb_ent = 'data/FB13/entities.txt'
fb_path = 'data/fb_question_rules.csv'
fb_rel = 'data/FB13/relations.txt'
fb_ent_text = 'data/FB13/entity2text.txt'

ent2id = {}
id2ent = {}
entities = []
ent2text = {}

with open(fb_ent, 'r') as e:
    lines = e.readlines()
    for j in lines:
        id, text = j.strip('\n').split('\t')
        entities.append(text)
        ent2id[text] = id
        id2ent[id] = text

# print(id2ent)
with open(fb_ent_text, 'r', encoding='utf-8') as e:
    lines = e.readlines()
    for j in lines:
        ent, text = j.strip('\n').split('\t')
        ent2text[ent] = text



rel2id = {}
id2rel = {}
with open(fb_rel, 'r') as e:
    lines = e.readlines()
    for j in lines:
        id, text = j.strip('\n').split('\t')
        rel2id[text] = id
        id2rel[id] = text


def is_question_in_kb(sentence):

    for i in entities:
        if i.replace('_',' ') in sentence.lower():
            return True
    return False

def check_exist_entity(head):
    head_word = head.lower().split()
    found = False
    f = ''
    for e in ent2id.keys():
        words = e.split('_')
        text = ent2text[e]
        if len(words) == len(head_word):
            for w in range(len(words)):
                if words[w] == head_word[w]:
                    found = True
                else:
                    found = False
            if found == True:
                return ent2id[e]
            else:
                if head in text:
                    return ent2id[e]
        else:
            if head in text:
                return ent2id[e]
    return -1


from gensim.models import Word2Vec
import pickle

def get_neighbor_nodes(G, kdtree, node_embeddings, node_id):
    # K-NN query
    degrees = G.degree()
    NodeDegree = []
    NodeName = []
    # Print the embeddings for each node
    data = []
    NodeLabel = []
    points = []
    # Print the node degrees
    for node, degree in degrees:
        NodeName.append(node)


    print('node id', node_id)
    print(degrees[str(node_id)])
    k = degrees[str(node_id)]
    query_point = node_embeddings.get_vector(str(node_id))
    distances, indices = kdtree.query(tuple(query_point), k=k)
    print('Neighbor nodes', indices)
    for node, degree in degrees:
    #     print(f"Node {node}: Degree {degree}")
    #     NodeDegree.append(degree)
        NodeName.append(node)
    #     embedding = node_embeddings.get_vector(str(node))
    #     # print(f"Node: {node}, Embedding: {embedding}")
    #     data.append(tuple(embedding))
    #     points.append(tuple(embedding))
    #     NodeLabel.append(node)
    nn_nodes = []
    for j in range(0, len(indices)):
        neighbor_index = indices[j]
        neighbor_name = NodeName[neighbor_index]
        nn_nodes.append(id2ent[neighbor_name])
        print('neighbor name', neighbor_name, id2ent[neighbor_name] )
        # print('neighbor name', neighbor_name, id2ent[str(neighbor_index)])

    return nn_nodes
    #
    # print("Print the nearest neighbors for each query point")
    # scores = []
    # len_points = len(points)
    # hit_score = []
    # for i in range(len(points)):
    #     print("\n=========================================")
    #     query_point = points[i]
    #     query_point_label = NodeLabel[i]
    #
    #     print("Node Label of this point", query_point_label)
    #     vt = NodeName.index(query_point_label)
    #     print("Degree:", NodeDegree[vt])
    #     k = NodeDegree[vt] + 1  # Number of nearest neighbors to retrieve
    #     distances, indices = kdtree.query(points, k=k)
    #     nearest_neighbors_indices = indices[i]
    #     nearest_neighbors_distances = distances[i]
    #     print(f"Nearest neighbors for point {query_point}:")
    #     print(f"Nearest neighbors for point index {nearest_neighbors_indices}:")
    #
    #     n_ls = []
    #     for j in range(1, len(nearest_neighbors_indices)):
    #         neighbor_index = nearest_neighbors_indices[j]
    #         neighbor_name = NodeLabel[nearest_neighbors_indices[j]]
    #         neighbor_distance = nearest_neighbors_distances[j]
    #         print(f'Node name: {neighbor_name} -- distance: {neighbor_distance}')
    #         n_ls.append(NodeLabel[nearest_neighbors_indices[j]])
    #     score_ls = nearest_neighbors_distances[1:].tolist()
    #     scores.append(1.0 / (score_ls.index(max(score_ls)) + 1))
    #     hit_score.append(score_ls.index(max(score_ls)) + 1 / len_points)
    #     print(f'Neighbor Node Name {set(n_ls)} -- {np.mean(nearest_neighbors_distances[1:])}')


# print(G.number_of_edges())
def extract_subgraph(graph, relation):
    subgraph = nx.Graph()
    for u, v, data in graph.edges(data=True):
        if 'relation' in data and data['relation'] == relation:
            subgraph.add_edge(u, v, **data)  # Copy edge attributes
    return subgraph


candidate = []
with open(fb_path, 'r') as f:
    data = csv.reader(f, delimiter='\t')
    count = 0
    for i in data:
        ans = i[1]
        ans_id = ent2id[ans.lower().replace(' ', '_')]
        q_encode = s_model.encode(i[0])
        if 'X' in i[3] and is_question_in_kb(i[3]) == True:
                triples = i[3].split('^')
                tail_X = []
                head_X = []
                non_X = []
                for triplet in triples:
                    print(triplet)
                    h, r, t = triplet.split('\t')

                    if t == 'X':
                        if '-' in r:
                            id_rel = rel2id[r[1:]]
                        else:
                            id_rel = rel2id[r]

                        head_id = check_exist_entity(h)
                        if int(head_id) > -1:
                            count += 1
                            print(h, '--tail X--', head_id)
                            with open(f'data/query_fb13/fb13/knowledge_graph_{id_rel}.pkl', 'rb') as file:
                                G = pickle.load(file)

                            with open(f'data/query_fb13/fb13/kd_tree_{id_rel}.pkl', 'rb') as file:
                                kd_tree = pickle.load(file)
                            m = Word2Vec.load(f'data/query_fb13/fb13/fb_kg_node2vec_{id_rel}.model')
                            node_embeddings = m.wv
                            try:
                                neighbor_nodes_tail = get_neighbor_nodes(G, kd_tree, node_embeddings, head_id)
                                print('answer---',i[1])
                                print('KD-tree nodes', neighbor_nodes_tail)
                                for node in neighbor_nodes_tail:
                                    print(f'triple--{h} {r} {node.replace("_", " ")}')
                                    tail_X.append(f'{id2ent[head_id]}\t{r}\t{node.replace("_", " ")}')
                                # break
                            except:
                                continue

                    elif h == 'X':
                        if '-' in r:
                            id_rel = rel2id[r[1:]]
                        else:
                            id_rel = rel2id[r]
                        tail_id = check_exist_entity(t)
                        if int(tail_id) > -1:
                            count+=1
                            print(t, '--head X--', tail_id)
                            with open(f'data/query_fb13/fb13/knowledge_graph_{id_rel}.pkl', 'rb') as file:
                                G = pickle.load(file)

                            with open(f'data/query_fb13/fb13/kd_tree_{id_rel}.pkl', 'rb') as file:
                                kd_tree = pickle.load(file)
                            m = Word2Vec.load(f'data/query_fb13/fb13/fb_kg_node2vec_{id_rel}.model')
                            node_embeddings = m.wv
                            try:
                                neighbor_nodes_head = get_neighbor_nodes(G, kd_tree, node_embeddings, tail_id)
                                print('answer--', i[1])
                                print('KD-tree nodes', neighbor_nodes_head)
                                for node in neighbor_nodes_head:
                                    print(f'triple--{node.replace("_", " ")} {r} {t}')
                                    head_X.append(f'{node.replace("_", " ")}\t{r}\t{id2ent[tail_id]}')
                                # break
                            except:
                                continue
                    else:
                        print('answer--',i[1])

                        non_h, non_r, non_t = triplet.split('\t')
                        if '-' in non_r:
                            non_r = non_r[1:]

                        non_h_id = check_exist_entity(non_h)
                        non_t_id = check_exist_entity(non_t)
                        if int(non_h_id) > -1 and int(non_t_id) > -1:
                            non_X.append(f'{id2ent[non_h_id]}\t{r}\t{id2ent[non_t_id]}')
                        elif int(non_h_id) > -1 and int(non_t_id) == -1:
                            with open(f'data/query_fb13/fb13/knowledge_graph_{rel2id[non_r]}.pkl', 'rb') as file:
                                G = pickle.load(file)

                            with open(f'data/query_fb13/fb13/kd_tree_{rel2id[non_r]}.pkl', 'rb') as file:
                                kd_tree = pickle.load(file)
                            m = Word2Vec.load(f'data/query_fb13/fb13/fb_kg_node2vec_{rel2id[non_r]}.model')
                            node_embeddings = m.wv
                            try:
                                neighbor_nodes_head = get_neighbor_nodes(G, kd_tree, node_embeddings, non_h_id)
                                # print('answer--', i[1])
                                print('KD-tree nodes', neighbor_nodes_head)
                                for node in neighbor_nodes_head:
                                    print(f'triple--{node.replace("_", " ")} {r} {t}')
                                    non_X.append(f'{id2ent[non_h_id]}\t{non_r}\t{node.replace("_", " ")}')
                                # break
                            except:
                                continue
                        elif int(non_h_id) == -1 and int(non_t_id) > -1:
                            with open(f'data/query_fb13/fb13/knowledge_graph_{rel2id[non_r]}.pkl', 'rb') as file:
                                G = pickle.load(file)

                            with open(f'data/query_fb13/fb13/kd_tree_{rel2id[non_r]}.pkl', 'rb') as file:
                                kd_tree = pickle.load(file)
                            m = Word2Vec.load(f'data/query_fb13/fb13/fb_kg_node2vec_{rel2id[non_r]}.model')
                            node_embeddings = m.wv
                            try:
                                neighbor_nodes_head = get_neighbor_nodes(G, kd_tree, node_embeddings, non_t_id)
                                # print('answer--', i[1])
                                print('KD-tree nodes', neighbor_nodes_head)
                                for node in neighbor_nodes_head:
                                    print(f'triple--{node.replace("_", " ")} {r} {t}')
                                    non_X.append(f'{node.replace("_", " ")}\t{non_r}\t{id2ent[non_t_id]}')
                                # break
                            except:
                                continue
                # if len(head_X) > 0 or len(tail_X) > 0:
                candidate.append([i[0], i[1], head_X, tail_X, non_X])

        print('-------------------')
# print(count)
# print(id2ent)
# print(candidate)
# print(len(candidate))

answer = []
for c in candidate:
    q = c[0]
    q_encode = s_model.encode(q)
    ground_truth = c[1]
    head = c[2]
    tail = c[3]
    non = c[4]
    candi_answer = []
    if len(head) > 0:
        for h in head:
            h_tem = ''
            t_tem = ''
            max_score = 0.0
            non_tem = []
            if len(tail) > 0:
                for t in tail:
                    if len(non) > 0:
                        ans = h.replace('\t',' ') + ' ' + t.replace('\t', ' ') + ' '.join([n.replace('\t', ' ') for n in non])
                        ans_encode = s_model.encode(ans)
                        score = util.cos_sim(q_encode, ans_encode)
                        if score > max_score:
                            h_tem = h
                            t_tem = t
                            max_score = score
                            non_tem = non

                    else:
                        ans = h.replace('\t', ' ') + ' ' + t.replace('\t', ' ')
                        ans_encode = s_model.encode(ans)
                        score = util.cos_sim(q_encode, ans_encode)
                        if score > max_score:
                            h_tem = h
                            t_tem = t
                            max_score = score
                            non_tem = []
                candi_answer.append([q, h_tem, t_tem, non_tem, score.detach().numpy()[0][0]])
            else:
                if len(non) > 0:
                    ans = h.replace('\t', ' ') + ' '.join([n.replace('\t', ' ') for n in non])
                    ans_encode = s_model.encode(ans)
                    score = util.cos_sim(q_encode, ans_encode)
                    if score > max_score:
                        h_tem = h
                        t_tem = []
                        max_score = score
                        non_tem = non
                else:
                    ans = h.replace('\t', ' ')
                    ans_encode = s_model.encode(ans)
                    score = util.cos_sim(q_encode, ans_encode)
                    if score > max_score:
                        h_tem = h
                        t_tem = []
                        max_score = score
                        non_tem = []
            candi_answer.append([q, h_tem, t_tem, non_tem, score.detach().numpy()[0][0]])


    elif len(tail) > 0:
        tem = ''
        max_score = 0.0
        non_tem = []
        for t in tail:
            if len(non) > 0:
                ans = t.replace('\t', ' ') + ' '.join([n.replace('\t', ' ') for n in non])
                ans_encode = s_model.encode(ans)
                score = util.cos_sim(q_encode, ans_encode)
                if score > max_score:
                    tem = t
                    max_score = score
                    non_tem = non

            else:
                ans = t.replace('\t', ' ')
                ans_encode = s_model.encode(ans)
                score = util.cos_sim(q_encode, ans_encode)
                if score > max_score:
                    tem = t
                    max_score = score
                    non_tem = []

        candi_answer.append([q, [], tem, non_tem, score.detach().numpy()[0][0]])

    else:
        ans = ' '.join([n.replace('\t', ' ') for n in non])
        ans_encode = s_model.encode(ans)
        score = util.cos_sim(q_encode, ans_encode)
        if score > 0.5:
            candi_answer.append([q, [], [], non, score.detach().numpy()[0][0]])

    # q_encode = s_model.encode(q)

    # cos_sim = util.cos_sim(q_encode, a_encode)
    #
    # all_sentence_combinations = []
    # for i in range(len(cos_sim) - 1):
    #     for j in range(i + 1, len(cos_sim)):
    #         all_sentence_combinations.append([cos_sim[i][j], i, j])
    #
    # # Sort list by the highest cosine similarity score
    # all_sentence_combinations = sorted(all_sentence_combinations, key=lambda x: x[0], reverse=True)
    #
    # print("Top-5 most similar pairs:")
    # for score, i, j in all_sentence_combinations[0:5]:
    #     print("{} \t {} \t {:.4f}".format(sentences[i], sentences[j], cos_sim[i][j]))
    print(candi_answer)
    answer.append([ground_truth, candi_answer])

# for ans in answer:
#     print(ans)
'''
qa_model = pipeline("question-answering")

for ans in answer:
    print(ans)
    rule_entity = []
    print(ans[1])
    if len(ans[1]) > 3:
        truth_answer = ans[0]
        question = ans[1][0][0]
        head = ans[1][0][1]
        tail = ans[1][0][2]
        non_item = ans[1][0][3]

        if len(head) > 0:
            print(head)
            ent1, _, ent2  = head.split('\t')
            if int(check_exist_entity(ent1)) > -1:
                rule_entity.append(ent1)
            if int(check_exist_entity(ent2)) > -1:
                rule_entity.append(ent2)
        if len(tail) > 0:
            ent3, _, ent4 = tail.split('\t')
            if int(check_exist_entity(ent3)) > -1:
                rule_entity.append(ent3)
            if int(check_exist_entity(ent4)) > -1:
                rule_entity.append(ent4)
        if len(non_item) > 0:
            for i in non_item:
                e1, _, e2 = i.split('\t')
                if int(check_exist_entity(e1)) > -1:
                    rule_entity.append(e1)
                if int(check_exist_entity(e2)) > -1:
                    rule_entity.append(e2)
    context = []
    rule_entity = list(set(rule_entity))
    print('rule entity ', rule_entity)
    if len(rule_entity) > 0:
        for r in rule_entity:
            try:
                context.append(ent2text[r.lower().replace(' ','_')])
            except:
                continue
        context = '. '.join(context)
        print(context)
        rs = qa_model(question=question, context=context)
        print(truth_answer, rs)
    else:
        print('Cannot find answer')

print(len(answer))

# for a in answer:
#     truth = a[0]
#     head = a[1][0][1]
#     tail = a[1][0][2]
#     non = a[1][0][3]
#     score = a[1][0][4]
#     print(f'{truth}--{head}--{tail}--{non}--{score}')
#     if truth in head:
#         h, _, _ = head.split('\t')
#         print('aaa', h)
#     elif truth in tail:
#         _,_, t = tail.split('\t')
#         print('bbb', t)

'''