#!/usr/bin/bash

url='https://1drv.ms/u/s!'AiohV3HRf-34jIdRaoWALpKnVJgkjw
name=tdn_sthv2_r50_8x1x1.pth

parsed=$(python -c "from workoutdetector.scripts.download import parse_onedrive; \
    print(parse_onedrive('$url'))")
echo parsed link: $parsed
printf "==> Download weights from OneDrive and save to checkpoints/finetune/$name[y/N]? "
read answer
echo $answer
if [ "$answer" != "${answer#[Yy]}" ]; then
    echo "Downloading weights from OneDrive..."
    wget $parsed -O checkpoints/finetune/$name
    echo "Done."
else
    echo "Skipped."
fi
