import pickle


def save_to_pickle(o, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(o, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_from_pickle(filename):
    with open(filename, 'rb') as handle:
        o = pickle.load(handle)
    return o

