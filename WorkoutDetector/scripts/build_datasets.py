import os

repcount_class = []
base = os.path.join(os.path.expanduser("~"), 'projects', 'WorkoutDetector')
with open(os.path.join(base, 'datasets/RepCount/classes.txt')) as f:
    for line in f:
        repcount_class.append(line.strip())

countix_class = []
with open(os.path.join(base, 'datasets/Countix/classes.txt')) as f:
    for line in f:
        countix_class.append(line.strip())

haa500_class = []
with open(os.path.join(base, 'datasets/haa500/classes.txt')) as f:
    for line in f:
        haa500_class.append(line.strip())


classes = [
    'front_raise', 'pull_up', 'squat', 'bench_pressing', 'jumping_jack', 'situp',
    'push_up', 'battle_rope', 'exercising_arm', 'lunge', 'mountain_climber',
    'shoulder_press', ]

repcount_to = [
    'front_raise', 'pull_up', 'squat', 'bench_pressing', 'jumping_jack', 'situp',
    'push_up', 'battle_rope']
repcount_d = {}
for i, x in enumerate(repcount_class):
    if x in repcount_to:
        repcount_d[i] = classes.index(x)

countix_to = [
    'exercising_arm', 'bench_pressing', 'front_raise', 'squat', 'jumping_jack', 'lunge',
    'mountain_climber', 'pull_up', 'push_up', 'situp']

countix_d = dict()
for i, x in enumerate(countix_class):
    y = countix_to[i]
    j = classes.index(y)
    countix_d[i] = j

print(repcount_d)
print(countix_d)

# write the new labels to file


def build(label_map, prefix, input_txt, output_txt):
    set_type = input_txt.split('/')[-1].split('.')[0]
    with open(input_txt, 'r') as f:
        lines = f.readlines()
    with open(output_txt, 'w') as f:
        for line in lines:
            path, length, label = line.rstrip().split()
            path, length, label = line.split()
            label = int(label)
            if label in label_map:
                label = label_map[label]
            else:
                continue
            path = '/'.join([prefix,set_type, path])
            f.write(f'{path} {length} {label}\n')


# new large dataset will be in path: data/Workouts
build(repcount_d,
      'RepCount',
      os.path.join(base, 'data/RepCount/rawframes/train.txt'),
      os.path.join(base, 'data/Workouts/rawframes/train_repcount.txt'))
build(repcount_d,
      'RepCount',
      os.path.join(base, 'data/RepCount/rawframes/val.txt'),
      os.path.join(base, 'data/Workouts/rawframes/val_repcount.txt'))
build(repcount_d,  'RepCount',
      os.path.join(base, 'data/RepCount/rawframes/test.txt'),
      os.path.join(base, 'data/Workouts/rawframes/test_repcount.txt'))
build(countix_d, 'Countix',
      os.path.join(base, 'data/Countix/rawframes/train.txt'),
      os.path.join(base, 'data/Workouts/rawframes/train_countix.txt'))
build(countix_d, 'Countix',
      os.path.join(base, 'data/Countix/rawframes/val.txt'),
      os.path.join(base, 'data/Workouts/rawframes/val_countix.txt'))

# Merge files
# cat train_repcount.txt train_countix.txt > train.txt
# cat val_repcount.txt val_countix.txt > val.txt
# cat test_repcount.txt > test.txt

# link to Workouts
# ln -s data/RepCount/rawframes/train data/Workouts/rawframes/RepCount/train
# ln -s data/RepCount/rawframes/val data/Workouts/rawframes/RepCount/val
# ln -s data/RepCount/rawframes/test data/Workouts/rawframes/RepCount/test
# ln -s data/Countix/rawframes/train data/Workouts/rawframes/Countix/train
# ln -s data/Countix/rawframes/val data/Workouts/rawframes/Countix/val