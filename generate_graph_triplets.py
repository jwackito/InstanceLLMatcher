import pandas as pd
import rdflib
from src.converter import create_graph_real_estate
from joblib import delayed, Parallel


df = pd.read_csv('../../InstanceLLMatcher/data/inmo_2024-02-01_23-41-27.csv', dialect='excel', keep_default_na=False, dtype=str)

def write_graph_for_line(line_number):
    g = create_graph_real_estate(df.iloc[line_number])
    with open(f'../../InstanceLLMatcher/output/triples/{line_number:06}.tsv', 'w') as f:
        f.write('subject\tpredicate\tobject\n')
        for s, p, o in g.triples((None, None, None)):
            f.write(f'{s.replace("\t", "_")}\t{p.replace("\t", "_")}\t{o.replace("\t", "_")}\n')
    print(line_number, end='\r')

Parallel(backend='multiprocessing', n_jobs=32)(delayed(write_graph_for_line)(i) for i in range(2000))
