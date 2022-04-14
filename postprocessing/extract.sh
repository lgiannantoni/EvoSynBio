#!/bin/bash
# Estrae la prima e l'ultima immagine di ogni protocollo simulato.
# Uso: ./extract.sh <cartella_esperimento>
# esempio: ./extract.sh ./half_n_half

ROOT_DIR=$PWD

if [[ $# -eq 0 ]]; then
	echo "Usage: ./extract.sh <experiment folder> [<experiment folder> ...]"
	exit
fi

for i in $@; do

	if [[ -d $i ]]; then
		EXPDIR=$i
		echo $EXPDIR

		RESULTS_DIR="$ROOT_DIR/$i"/results
		EXTRACT_DIR="$ROOT_DIR/$i"/extracted
		START_IMG_DIR="$EXTRACT_DIR"/start_images
		END_IMG_DIR="$EXTRACT_DIR"/end_images

		#cleanup and make new dirs
		if [[ -d $EXTRACT_DIR ]]; then
			rm -rf "$EXTRACT_DIR"
		fi	
		mkdir "$EXTRACT_DIR"
		mkdir "$START_IMG_DIR"
		mkdir "$END_IMG_DIR"

		# get images
		cd "$RESULTS_DIR"
		cp fitness.txt "$EXTRACT_DIR"
		for protocol_dir in $(ls . | grep -v fitness); do
			tokens=( $(echo $protocol_dir | tr '_' ' ') )
			protocol_name=${tokens[1]}

			#enter protocol directory
			cd ./$protocol_dir/GridSimulator
			LIST=( $(ls | sort -n | tr '\n' ' ') )
			#echo $LIST
			START_IMG=${LIST[0]}
			END_IMG=${LIST[-1]}

			if [[ -f $START_IMG && -f $END_IMG ]]; then
				cp ${START_IMG} "$START_IMG_DIR/$protocol_name".png
				cp ${END_IMG} "$END_IMG_DIR/$protocol_name".png
			fi
			cd ../..

		done

	else
		echo "Directory $i not found in $PWD"
	fi
done
