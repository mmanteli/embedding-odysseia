import pickle
from sqlitedict import SqliteDict
import sys

def yield_from_pickle(f_name):
    with open(f_name,"rb") as f:
        while True:
            try:
                dicts=pickle.load(f)
                yield dicts
            except EOFError:
                break

register = sys.argv[1]
assert register in ["dtp", "ne", "OP", "SP"]
db = SqliteDict(f"{register}.sqlite")
input_file = f"/scratch/project_462000883/amanda/embedded-data/e5/{register}_test.pkl"

for i, beet in enumerate(yield_from_pickle(input_file)):
    for b in beet:
        db[b["id"]+"-"+str(b["offset"])] = b
    if i%10000==0:   # not optimized, but results in at least okay performance
        db.commit()
db.commit()


