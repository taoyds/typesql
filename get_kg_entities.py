# -*- coding: utf-8 -*-
import json
import sys
import urllib
import torch
from typesql.utils import *
import numpy as np
import datetime
from nltk import ngrams

LOW_CHAR = ["a","aboard",
"about",
"above",
"across",
"after",
"against",
"along",
"amid",
"among",
"an",
"and",
"anti",
"around",
"as",
"at",
"before",
"behind",
"below",
"beneath",
"beside",
"besides",
"between",
"beyond",
"but",
"by",
"concerning",
"considering",
"despite",
"down",
"during",
"except",
"excepting",
"excluding",
"following",
"for",
"from",
"in",
"inside",
"into",
"like",
"minus",
"near",
"of",
"off",
"on",
"onto",
"opposite",
"or",
"outside",
"over",
"past",
"per",
"plus",
"regarding",
"round",
"save",
"since",
"so",
"than",
"the",
"through",
"to",
"toward",
"towards",
"under",
"underneath",
"unlike",
"until",
"up",
"upon",
"versus",
"via",
"with",
"within",
"without",
"yet"]

VISITED = []
#api_key = open('.api_key').read()
api_key = sys.argv[1]
sql_path = sys.argv[2]
service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
#sql_path = "data/dev_tok.jsonl"



def load_data(sql_path):
    sql_data = []
    print "Loading data from %s"%sql_path
    with open(sql_path) as inf:
        for idx, line in enumerate(inf):
            sql = json.loads(line.strip())
            sql_data.append(sql)

    return sql_data


def query_kg(query):
    global VISITED
    entity = None
    params = {
     'query': query,
     'limit': 1,
     'indent': True,
     'key': api_key,
    }
    if query not in VISITED:
        try:
            url = service_url + '?' + urllib.urlencode(params)
            response = json.loads(urllib.urlopen(url).read())
            element = response['itemListElement'][0]
            res = element['result']['name'].lower()
            #ent_type = [e.lower() for e in element['result']['@type']]
            try:
                ent_type = [e.lower() for e in element['result']['@type']]
                #desc = element['description']
            except:
                print "can not get ent_type!"
                ent_type = []
                #desc = ""

            if query == res:
                entity = (query.split(" "), ent_type)
        except:
            VISITED.append(query)
            VISITED = list(set(VISITED))
            pass

    return entity


sql_data = load_data(sql_path)
count = 0
sql_data_kg = []

for sql in sql_data:
    ents_kg = []
    question = sql["question"]
    q_toks = sql["question_tok"]
    q_sp = sql["question"].split(" ")
    upw_count = 3 #= len([q for q in q_sp if len(q) != 0 and q[0].isupper()])
    flag_3 = True
    flag_4 = True
    flag_5 = True

    #always run
    if upw_count > 2:
        #print "sentence is: ", sql["question"]
        #print "question_tok is: ", sql["question_tok"]
        bigrams = [q for q in ngrams(q_toks, 2) if q[0] not in LOW_CHAR]
        for bg in bigrams:
            qb = " ".join(bg)
            ent = query_kg(qb)
            if ent:
                ents_kg.append(ent)
        if flag_3:
            trigrams = [q for q in ngrams(q_toks, 3) if q[0] not in LOW_CHAR]
            for tg in trigrams:
                qt = " ".join(tg)
                ent = query_kg(qt)
                if ent:
                    ents_kg.append(ent)
        if flag_4:
            grams4 = [q for q in ngrams(q_toks, 4) if q[0] not in LOW_CHAR]
            for fg in grams4:
                qf = " ".join(fg)
                ent = query_kg(qf)
                if ent:
                    ents_kg.append(ent)
        if flag_5:
            grams5 = [q for q in ngrams(q_toks, 5) if q[0] not in LOW_CHAR]
            for fg in grams5:
                qf = " ".join(fg)
                ent = query_kg(qf)
                if ent:
                    ents_kg.append(ent)
    #else:
    #    print "skipping one---: ", sql["question"]

    sql["kg_entities"] = ents_kg
    if len(ents_kg) != 0:
        try:
            print "sentence is: ", sql["question"]
            print "kg_entities is: ", sql["kg_entities"]
        except:
            pass
    sql_data_kg.append(sql)
    count += 1

print "\nloaded data exmple count: ", count

out_dir = sql_path + ".kg"
with open(out_dir,'w') as wf:
    for d in sql_data_kg:
        wf.write(json.dumps(d) + "\n")

print "done!!!"

