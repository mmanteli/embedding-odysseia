import pickle
from sqlitedict import SqliteDict


def yield_from_pickle(f_name):
    with open(f_name,"rb") as f:
        while True:
            try:
                dicts=pickle.load(f)
                yield dicts
            except EOFError:
                break

db = SqliteDict("SP.sqlite")
input_file = "/scratch/project_462000883/amanda/embedded-data/e5/SP_test.pkl"

for i, beet in enumerate(yield_from_pickle(input_file)):
    for b in beet:
        db[b["id"]+"-"+str(b["offset"])] = b
    if i%10000==0:
        db.commit()
db.commit()


