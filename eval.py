from eval_lib import get_ap, evaluate_detections
import pickle
import argparse

parser = argparse.ArgumentParser(description='Phrase Detections Eval')
parser.add_argument('--dets_path', default='dets.pkl', type=str)
parser.add_argument('--dataset', default='refcoco+', type=str)
parser.add_argument('--set', default='test', type=str)
args = parser.parse_args()


with open('data/%s/%s/phrases.pkl'%(args.dataset, args.set), 'rb') as handle:
    phrases = pickle.load(handle)

with open('data/%s/%s/roidb.pkl'%(args.dataset, args.set), 'rb') as handle:
    roidb = pickle.load(handle)

with open('data/%s/%s/phrase_to_ind.pkl'%(args.dataset, args.set), 'rb') as handle:
    phrase_to_ind = pickle.load(handle)

with open('data/%s/%s/train_counts.pkl'%(args.dataset, args.set), 'rb') as handle:
    train_counts = pickle.load(handle)

with open(args.dets_path, 'rb') as handle:
    all_boxes = pickle.load(handle)

phrase_start = 0 
phrase_end = len(phrases)

all_ap, all_phrase_counts, all_top1acc, all_total_aug, all_top1acc_aug = get_ap(all_boxes, phrase_start, phrase_end, 
                                                                                phrase_to_ind, phrase_end, roidb, 
                                                                                phrases, './', filter_list =[])

evaluate_detections(all_ap, all_phrase_counts, all_top1acc, all_total_aug, all_top1acc_aug, phrase_to_ind, train_counts)
