import pickle
import pprint


label_file='src/kz_labels'
def read_label(src):
    pkl_file=open('src/kz_labels','rb')
    label = pickle.load(pkl_file)
    pprint.pprint(label)
    pkl_file.close


def get_label_dict():
    f=open('src/kz_labels','rb')
    label_dict = pickle.load(f)
    f.close()
    return label_dict

label_dict=get_label_dict()

print(label_dict)