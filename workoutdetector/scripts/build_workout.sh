# python scripts/build_datasets.py build_workout 

cd data/Workout/rawframes

cat train_repcount.txt train_countix.txt > train.txt
cat val_repcount.txt val_countix.txt > val.txt
cat test_repcount.txt > test.txt

cd ../../../

mkdir data/Workout/rawframes/RepCount/
mkdir data/Workout/rawframes/Countix/

ln -s data/RepCount/rawframes/train data/Workout/rawframes/RepCount/train
ln -s data/RepCount/rawframes/val data/Workout/rawframes/RepCount/val
ln -s data/RepCount/rawframes/test data/Workout/rawframes/RepCount/test
ln -s data/Countix/rawframes/train data/Workout/rawframes/Countix/train
ln -s data/Countix/rawframes/val data/Workout/rawframes/Countix/val

echo "Done building workout dataset."