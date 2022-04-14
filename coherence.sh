#!/bin/bash
[ $# -eq 0 ] && { echo "Usage: $0 dir-name"; exit 1; }

root_dir=$PWD
working_dir="$1"

if [ $# -eq 2 ]; then
  venv="$2"
else
  venv="venv"
fi

if ! [ -d "$working_dir" ]
then
    echo "Error: Directory $working_dir does not exist."
    exit
fi

if [ -d "$root_dir/$venv" ]; then
  v="$root_dir/$venv"
elif [ -d "$working_dir/$venv" ]; then
  v="$working_dir/$venv"
else
  echo "Error: Virtual environment $venv does not exist."
  exit
fi

echo "Activating python virtual environment $v"
source "$v/bin/activate" || { echo "Error: Could not activate $venv."; exit; }
echo "Changing current working directory to $working_dir"
cd "$working_dir" || { echo "Error: Could not cd to $working_dir."; exit; }
ugp3
echo "Done."
