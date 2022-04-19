import glob
import os.path as osp
import json
import numpy as np
from tqdm import tqdm
import logging
import subprocess
import re
import pickle as pkl
import math
import random

OUTPUT_DATA_DIR = './data/CATER-GEN-v2'  # Change path to where the videos are stored
mode = 'explicit'  # ambiguous or explicit
dataset = 'CATER-GEN-v2'
MAX_TOT_VIDEOS = 30000
np.random.seed(42)
NUM_ROWS = 3
NUM_COLS = 3

Shape_to_Name = {'spl': 'snitch', 'sphere': 'sphere', 'cylinder': 'cylinder', 'cube': 'cube', 'cone': 'cone'}

def check_avi_broken(fpath, max_frame):
    """ Check if the AVI file is broken, i.e. does not have index. This
    indicates a video that was not fully rendered and must be ignored for the
    final training/testing. """
    if osp.exists(fpath + '.lock'):
        # For any properly rendered video, the lock file must be deleted.
        return True
    try:
        output = subprocess.check_output(
            'ffmpeg -i {}'.format(fpath), shell=True, stderr=subprocess.STDOUT,
            universal_newlines=True)
    except subprocess.CalledProcessError as exc:
        output = exc.output
    prog = re.compile('.*AVI without index.*', flags=re.DOTALL)
    if prog.match(output):
        return True
    total_frames = subprocess.check_output(
        'ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 {}'.format(
            fpath), shell=True)
    if max_frame > int(total_frames):
        return True
    return False

def read_data(scene_files):
    data = {}
    for scene_file in tqdm(scene_files, desc='Reading metadata'):
        try:
            with open(scene_file, 'r') as fin:
                metadata = json.load(fin)
            vid_name = osp.splitext(scene_file.replace(
                '/scenes/', '/videos/'))[0] + '.avi'
            # Ignore videos that were not rendered correctly
            max_frame = max([ii[-1] for i in metadata['movements'].values() for ii in i])
            if check_avi_broken(vid_name, max_frame):
                continue
            data[vid_name] = metadata
            if len(data) > MAX_TOT_VIDEOS:
                # Since we don't make a list of others, might as well stop here
                break
        except Exception as e:
            logging.error('Unable to read {} due to {}'.format(
                scene_file, e))
    return data

def sort_data_for_train_test_split(data):
    print('Keeping {} of these videos'.format(MAX_TOT_VIDEOS))
    assert MAX_TOT_VIDEOS <= len(data), 'Data does not contain enough elts.'
    data = list(data.items())[:MAX_TOT_VIDEOS]
    np.random.shuffle(data)
    cut_point = int(0.8 * len(data))
    return data[:cut_point], data[cut_point:]

def find_quadrant(x, y):
    if x >= 0 and y >= 0:
        return 'the first quadrant'
    elif x < 0 <= y:
        return 'the second quadrant'
    elif x < 0 and y < 0:
        return 'the third quadrant'
    elif x >= 0 > y:
        return 'the fourth quadrant'
    else:
        print('ERROR!')

def coordinate_2d(raw_x, raw_y, num_rows=NUM_ROWS, num_cols=NUM_COLS):
    if num_rows != NUM_ROWS or num_cols != NUM_COLS:
        raw_x *= num_cols * 1.0 / NUM_COLS
        raw_y *= num_rows * 1.0 / NUM_ROWS
    if -num_rows < raw_x <= 0:
        raw_x -= 1
    if -num_cols < raw_y <= 0:
        raw_y -= 1
    x, y = (int(math.ceil(raw_x)), int(math.ceil(raw_y)))
    return x, y

def coarse_attribute(id, data):
    num = random.choice(range(0, 4))
    rand_attr = random.sample([data[id]['size'], data[id]['color'], data[id]['material']], num)
    rand_attr.append(Shape_to_Name[data[id]['shape']])
    return 'the ' + ' '.join(rand_attr)

def main():
    scene_files = glob.glob(osp.join(OUTPUT_DATA_DIR, 'scenes/*.json'))
    data_cache_fpath = osp.join(OUTPUT_DATA_DIR, 'good_videos.pkl')
    if osp.exists(data_cache_fpath):
        print('Found pre-computed file of good rendered data {}'.format(
            data_cache_fpath))
        with open(data_cache_fpath, 'rb') as fin:
            data = pkl.load(fin)
        print('...Read.')
    else:
        data = read_data(scene_files)
        with open(data_cache_fpath, 'wb') as fout:
            pkl.dump(data, fout)
    print('Found {} good videos out of {}'.format(len(data), len(scene_files)))
    
    train_data, test_data = sort_data_for_train_test_split(data)

    for split in ['train', 'test']:
        if split == 'train':
            split_data = train_data
        else:
            split_data = test_data
        split_anno = {}

        for idx in range(len(split_data)):
            video_path = '/'.join(split_data[idx][0].split('/')[-2:])
            metadata = split_data[idx][1]
            movements = metadata['movements']
            objects = metadata['objects']
            anno = ''
            for sbj_name, item in movements.items():
                if item == []:
                    continue
                sbj_id = [i for i, x in enumerate(objects) if x['instance'] == sbj_name][0]
                action, obj_name, start_frame, end_frame = item[0]
                final_pos = objects[sbj_id]['locations'][str(len(objects[sbj_id]['locations']) - 1)]
                if mode == 'ambiguous':
                    sbj_anno = coarse_attribute(sbj_id, objects)
                    x = find_quadrant(final_pos[0], final_pos[1])
                elif mode == 'explicit':
                    sbj_anno = 'the {} {} {} {}'.format(objects[sbj_id]['size'], objects[sbj_id]['color'],
                                                        objects[sbj_id]['material'],
                                                        Shape_to_Name[objects[sbj_id]['shape']])
                    x1, y1 = coordinate_2d(final_pos[0], final_pos[1], 3, 3)
                    x = '({}, {})'.format(x1, y1)
                if dataset == 'CATER-GEN-v1':
                    sbj_anno = 'the {}'.format(Shape_to_Name[objects[sbj_id]['shape']])

                if action == '_slide':
                    anno = anno + ' {} is sliding to {}.'.format(sbj_anno, x)
                if action == '_rotate':
                    anno = anno + ' {} is rotating.'.format(sbj_anno)
                if action == '_pick_place':
                    anno = anno + ' {} is picked up and placed to {}.'.format(sbj_anno, x)
                if action == '_contain':
                    obj_id = [i for i, x in enumerate(objects) if x['instance'] == obj_name][0]
                    if mode == 'ambiguous':
                        obj_anno = coarse_attribute(obj_id, objects)
                    elif mode == 'explicit':
                        obj_anno = 'the {} {} {} {}'.format(objects[obj_id]['size'], objects[obj_id]['color'], objects[obj_id]['material'], Shape_to_Name[objects[obj_id]['shape']])
                    if dataset == 'CATER-GEN-v1':
                        obj_anno = 'the {}'.format(Shape_to_Name[objects[obj_id]['shape']])
                    anno = anno + ' {} is picked up and containing {}.'.format(sbj_anno, obj_anno)

            split_anno[idx] = {
                'video': video_path,
                'caption': anno
            }
        with open(osp.join(OUTPUT_DATA_DIR, '{}_{}.json'.format(split, mode)), 'w') as fp:
            json.dump(split_anno, fp)


if __name__ == '__main__':
    main()
