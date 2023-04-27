
import argparse
import os
import json
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import spacy

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--data-root', default='../datasets/CATER-GEN-v2')
parser.add_argument('--gen-caption', default='./models/MAGE+/catergenv2_diverse/videos/generated_captions.json')
parser.add_argument('--mode', default='ambiguous', type=str, help='explicit or ambiguous')

def test_metrics_offline():
    args = parser.parse_args()

    with open(args.gen_caption, 'r') as fp:
        gen_captions = json.load(fp)
    gt_caption_path = os.path.join(args.data_root, f'test_{args.mode}.json')
    with open(gt_caption_path, 'r') as fp:
        gt_captions = json.load(fp)

    new_gt_captions = {}
    for key in gt_captions:
        video_id = gt_captions[key]['video']
        caption = gt_captions[key]['caption']
        new_gt_captions[os.path.basename(video_id)] = caption

    i = 0
    P_act, P_re = 0, 0
    for idx in range(len(gen_captions)):
        i += 1
        print(i)
        video_id = gen_captions[idx]["image_id"]
        video_id = video_id.split('.')[0] + '.avi'
        gen_caption = gen_captions[idx]["caption"]
        gt_caption = new_gt_captions[video_id]

        gt_parsing = sen_parse(gt_caption, mode=args.mode)
        gen_parsing = sen_parse(gen_caption, mode=args.mode)
        p_act, p_re = cross_check(gt_parsing, gen_parsing)
        P_act, P_re = P_act + p_act, P_re + p_re

    print('Action_Precision: ', P_act / i)
    print('Referring_Expression_Precision: ', P_re / i)

# python -m spacy download en_core_web_sm
Attributes = ['cone', 'snitch', 'sphere', 'cylinder', 'cube', 'small', 'medium', 'large', 'metal', 'rubber',
              'gold', 'gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
Quadrant = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4}
def sen_parse(text, mode='ambiguous'):
    parser = spacy.load('en_core_web_sm')  #, disable=['ner','textcat']
    sentences = sent_tokenize(re.sub(u'\\[SEP]|\\[CLS]', '', text)) # text.split('.')

    parsing = []
    for sen in sentences:
        doc = parser(sen)
        verbs = [tok for tok in doc if tok.pos_ == "VERB"]
        if len(verbs) == 0:
            sub_att = [tok for tok in word_tokenize(sen) if tok in Attributes]
            parsing.append({'subject': sub_att, 'motion': None, 'object': None})
        else:
            sub = sen[:sen.find(verbs[0].text)]
            sub_att = [tok for tok in word_tokenize(sub) if tok in Attributes]
            if verbs[0].text == 'rotating':
                motion = 'rotate'
                obj_att = None
            else:
                motion, obj_att = None, None
                obj = sen[sen.find(verbs[-1].text) + len(verbs[-1].text):]
                if verbs[-1].text == 'sliding':
                    motion = 'slide'
                    obj_att = find_quadrant(obj) if mode=='ambiguous' else find_coordinate(obj)
                elif verbs[-1].text == 'placed':
                    motion = 'pick-place'
                    obj_att = find_quadrant(obj) if mode=='ambiguous' else find_coordinate(obj)
                elif verbs[-1].text == 'containing':
                    motion = 'pick-contain'
                    obj_att = [tok for tok in word_tokenize(obj) if tok in Attributes]
            parsing.append({'subject': sub_att, 'motion': motion, 'object': obj_att})

    return parsing

def find_quadrant(text):
    if 'quadrant' in text:
        quadrant = [tok for tok in word_tokenize(text) if tok in Quadrant.keys()]
        return Quadrant[quadrant[0]]
    else:
        text = text.replace(' ', '')
        try:
            loc = text[text.find('(') + 1:text.find(')')]
            x, y = loc.split(',')
            x, y = int(x), int(y)
            if x >= 0 and y >= 0:
                quadrant = 1
            elif x < 0 <= y:
                quadrant = 2
            elif x < 0 and y < 0:
                quadrant = 3
            elif x >= 0 > y:
                quadrant = 4
        except:
            quadrant = None
        return quadrant

def find_coordinate(text):
    text = text.replace(' ', '')
    try:
        loc = text[text.find('(') + 1:text.find(')')]
        x, y = loc.split(',')
        x, y = int(x), int(y)
        coordinate = [x, y]
    except:
        coordinate = None
    return coordinate

def precision(gt, gen):
    tp_a, fp_a, tp_m, fp_m = 0, 0, 0, 0
    for attri in gt['subject']:
        if attri in gen['subject']:
            tp_a += 1
        else:
            fp_a += 1

    if gt['motion'] == gen['motion']:
        tp_m += 1
        if gt['motion'] in {'slide', 'pick-place'}:
            if gen['object'] is not None and gt['object'] == gen['object']:
                tp_m += 1
            else:
                fp_m += 1
        elif gt['motion'] in {'pick-contain'}:
            for attri in gt['object']:
                if gen['object'] is not None and attri in gen['object']:
                    tp_a += 1
                else:
                    fp_a += 1
    else:
        fp_m += 1

    return tp_a, fp_a, tp_m, fp_m

def cross_check(gt_list, gen_list):
    if len(gen_list) == 0:
        P_m, P_a = 0, 0
    elif len(gt_list) == 1 and len(gen_list) == 1:
        TP_a, FP_a, TP_m, FP_m = precision(gt_list[0], gen_list[0])
        P_m, P_a = TP_m / (TP_m + FP_m), TP_a / (TP_a + FP_a)
    else:
        if len(gen_list) == 1:
            tp_a0, fp_a0, tp_m0, fp_m0 = precision(gt_list[0], gen_list[0])
            tp_a1, fp_a1, tp_m1, fp_m1 = precision(gt_list[1], gen_list[0])
        elif len(gt_list) == 1:
            tp_a0, fp_a0, tp_m0, fp_m0 = precision(gt_list[0], gen_list[0])
            tp_a1, fp_a1, tp_m1, fp_m1 = precision(gt_list[0], gen_list[1])
        else:
            tp_a0, fp_a0, tp_m0, fp_m0 = tuple(map(sum, zip(precision(gt_list[0], gen_list[0]), precision(gt_list[1], gen_list[1]))))
            tp_a1, fp_a1, tp_m1, fp_m1 = tuple(map(sum, zip(precision(gt_list[1], gen_list[0]), precision(gt_list[0], gen_list[1]))))

        p_m0 = tp_m0 / (tp_m0 + fp_m0)
        p_a0 = tp_a0 / (tp_a0 + fp_a0)
        p_m1 = tp_m1 / (tp_m1 + fp_m1)
        p_a1 = tp_a1 / (tp_a1 + fp_a1)
        if p_m0 > p_m1:
            P_m, P_a = p_m0, p_a0
        elif p_m0 == p_m1 and p_a0 > p_a1:
            P_m, P_a = p_m0, p_a0
        else:
            P_m, P_a = p_m1, p_a1

    return P_m, P_a

if __name__ == '__main__':
    test_metrics_offline()
