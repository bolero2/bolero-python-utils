from uuid import uuid1, uuid4


def get_hash_value():
    id1 = str(uuid1()).split('-')[0]
    id2 = str(uuid4()).split('-')[0]

    return id1 + id2
