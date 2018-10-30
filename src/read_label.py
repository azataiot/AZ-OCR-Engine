import pickle
import pprint

pkl_file=open('src/kz_labels','rb')
label = pickle.load(pkl_file)
pprint.pprint(label)

pkl_file.close
