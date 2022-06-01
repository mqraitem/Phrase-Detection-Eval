import numpy as np 
from collections import Counter
from cython_bbox import bbox_overlaps

TEST_SUCCESS_THRESH = 0.5
TEST_PHRASE_COUNT_THRESHOLDS = np.array([0, 100, np.inf])

def get_phrase_predictions(im_boxes, gt, phrase, tokens, gt_scores, phrase_start, phrase_end, phrase_to_ind):
    phrase_index = phrase_to_ind[phrase]
    labels = None
    if phrase_index >= phrase_start and phrase_index < phrase_end:
        boxes = im_boxes[phrase_index - phrase_start]
        assert(boxes.shape[1] == 5)
        pred = boxes[:, :-1].reshape((-1, 4)).astype(np.float)
        overlaps = bbox_overlaps(pred, gt)
        labels = np.zeros(len(overlaps), np.float32)
        ind = np.where(overlaps >= TEST_SUCCESS_THRESH)[0]
        if len(ind) > 1:
            ind = min(ind)

        labels[ind] = 1
        # if tokens is None or len(self._phrase2token[phrase].intersection(tokens)) > 0:
        gt_scores[phrase_index] += list(boxes[:, -1])
        # else:
            # gt_scores[phrase_index] += list(np.ones(len(labels), np.float32) * -np.inf)

        assert(len(labels) == len(boxes))
    return phrase_index, labels

def get_ap(all_boxes, phrase_start, phrase_end, phrase_to_ind, num_phrases, roidb, processed_phrases, output_dir=None, filter_list =[]):
    """
    all_boxes is a list of length number-of-classes.
    Each list element is a list of length number-of-images.
    Each of those list elements is either an empty list []
    or a numpy array of detection.

    all_boxes[class][image] = [] or np.array of shape #dets x 5
    """
    # For each image get the score and label for the top prediction
    # for every phrase
    gt_scores = [[] for _ in range(num_phrases)]
    gt_labels = [[] for _ in range(num_phrases)]
    phrase_counts = Counter()
    top1acc = 0.
    total_aug = 0.
    top1acc_aug = 0.
    for index, im_boxes in enumerate(all_boxes):
        tokens = None

        roi = roidb[index]
        phrases_seen = list()
        i = 0
        for gt, phrase in zip(roi['boxes'], roi['processed_phrases']):
            if phrase in filter_list: 
                continue 
            phrase_index = phrase_to_ind[phrase]
            gt = gt.reshape((1, 4)).astype(np.float)
            seen, labels = get_phrase_predictions(im_boxes, gt, phrase, tokens, gt_scores, phrase_start, phrase_end, phrase_to_ind)
            if labels is not None:
                top1acc += labels[0]
                phrases_seen.append(seen)
                gt_labels[phrase_index] += list(labels)

        phrase_counts.update(phrases_seen)
        phrases_seen = set(phrases_seen)
        for phrase_index, boxes in zip(range(phrase_start, phrase_end), im_boxes):
            if phrase_index not in phrases_seen:
                # if tokens is None or len(self._phrase2token[processed_phrases[phrase_index]].intersection(tokens)) > 0:
                gt_scores[phrase_index] += list(boxes[:, -1])
                # else:
                    # gt_scores[phrase_index] += list(np.ones(len(boxes), np.float32) * -np.inf)
                gt_labels[phrase_index] += list(np.zeros(len(boxes), np.float32))

    # Compute average precision
    ap = np.zeros(num_phrases, np.float32)
    for phrase_index in range(phrase_start, phrase_end):
        phrase_labels = gt_labels[phrase_index]
        phrase_scores = gt_scores[phrase_index]
        order = np.argsort(phrase_scores)
        phrase_labels = np.array([phrase_labels[i] for i in order])
        pos_labels = np.where(phrase_labels)[0]
        n_pos = len(pos_labels)
        c = 0
        if n_pos > 0:
            # take into account ground truth phrases which were not
            # correctly localized
            n_missing = phrase_counts[phrase_index] - n_pos
            prec = [(n_pos - i) / float(len(phrase_labels) - index) for i, index in enumerate(pos_labels)]
            rec = [(n_pos - i) / float(n_pos + n_missing) for i, _ in enumerate(pos_labels)]
            c = np.sum([(rec[i] - rec[i+1])*prec[i] for i in range(len(pos_labels)-1)]) + prec[-1]*rec[-1]

            ap[phrase_index] = c

    return ap, phrase_counts, top1acc, total_aug, top1acc_aug

def evaluate_detections(ap, phrase_counts, top1acc, total_aug, top1acc_aug, phrase_to_ind, train_counts):
    """
    all_boxes is a list of length number-of-classes.
    Each list element is a list of length number-of-images.
    Each of those list elements is either an empty list []
    or a numpy array of detection.

    all_boxes[class][image] = [] or np.array of shape #dets x 5
    """
    # organize mAP by the number of occurrences
    count_thresholds = TEST_PHRASE_COUNT_THRESHOLDS
    mAP = np.zeros(len(count_thresholds))
    occurrences = np.zeros_like(mAP)
    samples = np.zeros_like(mAP)
    count_index = 0
    for phrase, phrase_index in phrase_to_ind.items():
        n_occurrences = phrase_counts[phrase_index]
        if n_occurrences < 1:
            continue

        train_count = 0
        if phrase in train_counts:
            train_count = train_counts[phrase]

        count_index = min(np.where(train_count <= count_thresholds)[0])
        mAP[count_index] += ap[phrase_index]
        occurrences[count_index] += 1
        samples[count_index] += n_occurrences

    mAP = mAP / occurrences
    thresh_string = '\t'.join([str(thresh) for thresh in count_thresholds])
    print('\nThresholds:  \t' + thresh_string + '\tOverall')

    ap_string = '\t'.join(['%.3f' % round(t * 100, 2) for t in mAP])
    print('AP:            \t' + ap_string + '\t%.3f' % round(np.mean(mAP) * 100, 2))

    occ_string = '\t'.join(['%i' % occ for occ in occurrences])
    print('Per Thresh Cnt:\t' + occ_string + '\t%i' % np.sum(occurrences))

    n_total = np.sum(samples)
    sample_string = '\t'.join(['%i' % item for item in samples])
    print('Instance Cnt:  \t' + sample_string + '\t%i' % n_total)

    acc = round((top1acc/(n_total - total_aug))*100, 2)
    print('Orig Localization Accuracy: %.2f' % acc)


    return np.mean(mAP)
