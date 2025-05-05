from sqlitedict import SqliteDict
import sys
db = SqliteDict("/scratch/project_462000883/amanda/embedding-extraction/dtp.sqlite")


def find_full_text(db, index):
    current_ind = 0  # they all start with 0
    t=""
    register=None
    while True:
        try:
            next_id = index+"-"+str(current_ind)
            d = db[next_id]
            t_new = d["text"]
            t += " "+t_new
            current_ind += len(t_new)   # next index is num_characters away
            if register is None:
                register = d["register"]
        except KeyError as e:
            #print(e)
            break
    return t, register


#print(len(db))


id_to_find = "5c808fedd393e4f586bb67bb3955c024"
print(find_full_text(db, id_to_find))

"""DTP
"5be7eabe95a0ee2f7653dd0f5f27e724"   #500k does not exist anymore
"5c808fedd393e4f586bb67bb3955c024"  # 200k exists
"bae0444728f600bd2c6e1241acad36e6"  # 100k exists
"4d34b0b66901fb4d302beeb689fef359"  # 50k

"""

"""SP
acd08c41898719c33162fab7fe926a6b   # 1M   # does not exist anymore
85c2c7ff5d9e891edd6490f4f27e76b5   # 500k  # does not exist anymore
d9418f1011d54947a51d94c281c1e384   # 250k  # does not exist any 
fa79034b1d6ae0d9508a4fd3f2c49628  # 100k # does not exist anymore
6b82c8e3f434a8466cfa6ae9d7941ec1  # 90k # does not exist anymore
dc50a7f155d07395def419c55effc5ef # 80k # does not exist anymore
f57f3676ccfd80b7ed54685b9094ca0c  # 79k
9c681a976e051cc9d2cd6c9cd9aa7432  # 78k
b77c0616e37dd77b79be26c1c15897b2  # 75k
cea27fdec95c5abf518551d546af3c74 # 70k
f703621882a602bfcfd5854e87f01db2   # 50k
abac7fff95c43500f3600c5b9c4ef624   # 10k
36c67381b235a458405cfa44ef7eda39  
"""
