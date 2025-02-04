import pandas as pd
import rdflib
import os

from csv2pronto.src.converter import create_graph_real_estate
from joblib import Parallel, delayed

df = pd.read_csv('data/inmo_2024-02-01_23-41-27.csv', dialect='excel', keep_default_na=False, dtype=str)

owl = rdflib.Namespace("http://www.w3.org/2002/07/owl#")
ns2 = rdflib.Namespace("https://w3id.org/rec/core/#")

sameAs = owl.sameAs


filenames = [x for x in os.scandir('output/InstanceLLMatcher/') if x.name.endswith('.log')]

lines = []
for x in filenames:
    lines.append(open(x,'r').readlines()[0])
lines = set(lines)

def create_graph_with_sameAs(line):
    query = """
        SELECT ?space WHERE {
            ?space a ns2:Space .
        }
    """
    g = rdflib.Graph()
    for idx in line:
        g += create_graph_real_estate(df.iloc[int(idx)])
    g.print()
    result = g.query(query)
    spaces = [row.space for row in result]
    for space1, space2 in zip(spaces, spaces[1:]):
        g.add((space1, sameAs, space2))
    return g

lines_train = list(lines)[:int(len(lines)*.8)]
lines_val = list(lines)[int(len(lines)*.8):]

graphs = Parallel(backend='multiprocessing', n_jobs=32)(delayed(create_graph_with_sameAs)(line.split()) for line in lines_train)
biggraph = rdflib.Graph()
for graph in graphs:
    biggraph += graph
biggraph.serialize('output/biggraph_with_sameAs_train.ttl')
with open(f'output/InstanceLLMatcher/triples/dataset_train.tsv', 'w') as f:
    for s, p, o in biggraph.triples((None, None, None)):
        f.write(f'{s.replace("\t", "_")}\t{p.replace("\t", "_")}\t{o.replace("\t", "_")}\n')

graphs = Parallel(backend='multiprocessing', n_jobs=32)(delayed(create_graph_with_sameAs)(line.split()) for line in lines_val)
biggraph = rdflib.Graph()
for graph in graphs:
    biggraph += graph
biggraph.serialize('output/biggraph_with_sameAs_val.ttl')
with open(f'output/InstanceLLMatcher/triples/dataset_validation.tsv', 'w') as f:
    for s, p, o in biggraph.triples((None, None, None)):
        f.write(f'{s.replace("\t", "_")}\t{p.replace("\t", "_")}\t{o.replace("\t", "_")}\n')

