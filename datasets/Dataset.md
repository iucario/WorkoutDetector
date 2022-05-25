# Datasets

Workout videos selected from Countix, haa500, RepCount.

Didn't check the overlap between the datasets.

## Workouts

A dataset merged from Countix and RepCount.

classes = [
    'front_raise', 'pull_up', 'squat', 'bench_pressing', 'jumping_jack', 'situp',
    'push_up', 'battle_rope', 'exercising_arm', 'lunge', 'mountain_climber']

Will add more fine-grained classes later.


## Countix

https://www.deepmind.com/open-source/kinetics

`workouts_train.csv` and `workouts_val.csv` are created from Deepmind's Countix dataset.

It contains 10 classes of physical training actions. With a total of 1092 samples from 1041 unique Youtube videos. Some samples are from the same video. The samples from the same video can be grouped together to a 10 second long video. Because the Countix dataset is from Kinetics 700.


| class | train |
| --- | --- |
| exercising arm | 112 |
| bench pressing | 59 |
| front raises | 131 |
| squat | 70 |
| jumping jacks | 117 |
| lunge | 122 |
| mountain climber (exercise) | 147 |
| pull ups | 115 |
| push up | 124 |
| situp | 95 |


### Missing Data

After cleanning there are 985 videos remained. val 332, train 975.

`yt-dlp` seems can not download videos with long height.

And some do not exist on Youtube now.


## RepCount

https://svip-lab.github.io/dataset/RepCount_dataset.html

Need careful cleaning.

There are typos in class names. And inconsistent class names. For example, `pullups` and `pull_up` are different classes in the provided csv file. Need to be merged.

| type | count_train | count_val |
| --- | --- | --- |
| battle_rope | 13 | 3 |
| bench_pressing | 93 | 13 |
| front_raise | 93 | 19 |
| jump_jack | 76 | 15 |
| others | 37 | 0 |
| pommelhorse | 69 | 15 |
| pull_up | 94 | 14 |
| push_up | 88 | 18 |
| situp | 93 | 18 |
| squat | 101 | 16 |


## haa500

https://www.cse.ust.hk/haa/

Contains videos of length 1 to 2 seconds.

battle-rope_wave
bench_dip
burpee
gym_lift
gym_lunges
gym_plank
gym_pull
gym_push
gym_ride
gym_run
jumping_jack
pushup
side_lunge
situp
workout_chest-pull
workout_crunch
