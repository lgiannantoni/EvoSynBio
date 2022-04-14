#!/bin/bash

coherence=./coherence.sh
exp_folder=$PWD
experiments=(
    circle
    half
    stripe
)
for experiment in "${experiments[@]}"; do
  if [[ ! -d ${experiment} ]];
  then
    echo "Directory ${experiment} does not exist."
    exit
  else
    echo "Launching experiment ${experiment} in detached screen..."
    screen -dmS "${experiment^^}" bash -c "cd ..; $coherence ${exp_folder}/${experiment} venv; sleep 1000d"
    echo "Experiment ${experiment} is running in screen ${experiment^^}."
  fi
done