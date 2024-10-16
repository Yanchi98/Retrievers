# import numpy as np
# from faiss import IndexFlatIP
#
# faiss_index = IndexFlatIP(64)
# testdata = np.random.random((5, 64))
# faiss_index.add(testdata)
# searchdata = np.random.random((1, 64))
# print(faiss_index.search(searchdata, 2))


from random import shuffle
queries = ["111","222","333"]
shuffle(queries)
print(queries)