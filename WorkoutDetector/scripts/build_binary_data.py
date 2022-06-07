from WorkoutDetector.utils.common import Repcount
import os
import shutil
import yaml
import tqdm


config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), '../utils/config.yml')))
base = config['proj_root']

class_name = 'squat'

repcount = Repcount()

def build_split(classname, split='train'):
    """Copy images to binary classification directory.
    
    |- data/Binary/squat
        |- train
            |- vid_1_start-frame
               |- img_00001.jpg
               |- img_00002.jpg
        |- val
           |- vid_2_start-frame

    start - mid: 0, mid - end: 1"""
    dst_data_dir = os.path.join(base, f'data/Binary/{classname}')
    df = repcount.get_anno(split)
    df = df[df['type'] == classname]
    data_root = os.path.join(repcount.data_root, 'rawframes', split)
    vids = df['name'].values
    
    # for vid in tqdm(vids):
    #     vid = vid.split('.')[0]
    #     count, reps = repcount.get_count(vid)
    #     for start, end in zip(reps[0::2], reps[1::2]):
    #         start += 1
    #         end += 1
    #         mid = (start + end) // 2
    #         os.mkdir(os.path.join(dst_data_dir, f'{vid}_{start}'))
    #         os.mkdir(os.path.join(dst_data_dir, f'{vid}_{mid}'))
    #         for i in range(start, mid):
    #             j = i-start+1
    #             src_path = os.path.join(data_root, f'{vid}/img_{i:05}.jpg')
    #             dst_path = os.path.join(dst_data_dir, f'{vid}_{start}/img_{j:05}.jpg')
    #             shutil.copy(src_path, dst_path)
    #         for i in range(mid, end):
    #             j = i-mid+1
    #             src_path = os.path.join(data_root, f'{vid}/img_{i:05}.jpg')
    #             dst_path = os.path.join(dst_data_dir, f'{vid}_{mid}/img_{j:05}.jpg')
    #             shutil.copy(src_path, dst_path)

    label_file = os.path.join(base, f'data/Binary/{classname}-{split}.txt')
    with open(label_file, 'w') as f:
        for vid in vids:
            vid = vid.split('.')[0]
            count, reps = repcount.get_count(vid)
            for start, end in zip(reps[0::2], reps[1::2]):
                start += 1
                end += 1
                mid = (start + end) // 2
                f.write(f'{classname}/{vid}_{start} {mid-start} 0\n')
                f.write(f'{classname}/{vid}_{mid} {end-mid} 1\n')


def rename_images():
    data_dir = os.path.join(base, f'data/Binary/{class_name}')
    for d in os.listdir(data_dir):
        for img in os.listdir(os.path.join(data_dir, d)):
            offset = int(d.split('_')[2])
            img_id = int(img.split('.')[0].split('_')[1])
            new_id = img_id - offset + 1
            new_name = f'{new_id:05}.jpg'
        for img in os.listdir(os.path.join(data_dir, d)):
            new_name = f'img_{img}'
            os.rename(os.path.join(data_dir, d, img), os.path.join(data_dir, d, new_name))


if __name__ == '__main__':
    build_split(class_name, split='train')
    build_split(class_name, split='val')
    build_split(class_name, split='test')
# rename_images()