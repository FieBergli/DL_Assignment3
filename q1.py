from data import load_imdb

(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)
#print([i2w[w] for w in x_train[141]])
print(w2i[".pad"])

