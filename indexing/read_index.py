from sqlitedict import SqliteDict
import sys
db = SqliteDict("/scratch/project_462000883/amanda/embedding-extraction/SP.sqlite")


def find_full_text(db, index):
    current_ind = 0  # they all start with 0
    t=""
    register=None
    while True:
        try:
            (current_ind)
            next_id = index+"-"+str(current_ind)
            d = db[next_id]
            t_new = d["text"]
            t += " "+t_new
            current_ind += len(t_new)
            if register is None:
                register = d["register"]
        except KeyError as e:
            #print(e)
            break
    return t, register


print(len(db))


id_to_find = "f2bfef7f04fa44ff5ab0d8ea646386ae"
print(find_full_text(db, id_to_find))


"""
36c67381b235a458405cfa44ef7eda39-0
36c67381b235a458405cfa44ef7eda39-99
36c67381b235a458405cfa44ef7eda39-192
36c67381b235a458405cfa44ef7eda39-212
36c67381b235a458405cfa44ef7eda39-281
36c67381b235a458405cfa44ef7eda39-332
36c67381b235a458405cfa44ef7eda39-373
36c67381b235a458405cfa44ef7eda39-425
"""
