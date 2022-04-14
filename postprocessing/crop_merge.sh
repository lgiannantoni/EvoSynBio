#!/bin/bash

#fa il crop delle immagini di GridSimulator (estrae il grafico a sinistra) e le monta per generazioni
#di individui (1 riga --> 1 generazione)
#si suppone che la cartella in input contenga 1 immagine per protocollo (es. lo step 0 della simulazione di ogni protocollo)
#il secondo argomento è il file statistics.csv prodotto in output da ugp
#il terzo è il file fitness.txt
#da questi due file viene derivata la struttura della matrice finale di immagini (ogni riga 1 generazione)
#uso: ./crop_merge.sh /.../cartella_immagini /.../statistics.csv
#esempio: ./crop_merge.sh half_n_half/extracted/start_images ./half_n_half/statistics.csv half_n_half/extracted/fitness.txt

ROOT_DIR=$PWD

if [[ $# -lt 3 ]]; then
	echo "Usage: <image folder> <statistics.csv> <fitness.txt>"
	exit
fi

WORKDIR="$1"
STATS=$(realpath "$2")
FITNESS=$(realpath "$3")

if [[ ! -d "$WORKDIR" ]]; then
	echo ...
	exit
fi

cd "$WORKDIR"

#crop images
CROPPED=./cropped
if [[ -d $CROPPED ]]; then
		rm -rf "$CROPPED"
fi	
mkdir "$CROPPED"
IND_MATRIX=individuals_matrix.txt

for i in $(ls *.png); do
	tokens=( $(echo $i | tr '.png' ' ') )
	protocol_name=${tokens[0]}
	#echo cropping $protocol_name
	#crop 140x140 patch (left chart)
	convert "$i" -crop 140x140+126+170 "$CROPPED/$protocol_name".png
done

sleep 10

cd cropped

#make individuals tmp file (1 row <--> 1 generation)
#n.b. prevale il file statistics.csv, quindi solo le generazioni già valutate per intero
#eventuali individui presenti in fitness.txt e facenti parte di una generazione non ancora valutata per intero,
#vengono ignorati
while read line; do
	#extract 9th column (cumulative #evaluations)
	EVAL_COL=$(cut -d ',' -f9)
	#echo $EVAL_COL

	PREV_EVAL=1
	LONGEST_ROW=0
	for n in $EVAL_COL; do
		#extract next n-PREV_EVAL lines from fitness file, get first column (protocol names)
		ROW=$(sed -n "$PREV_EVAL,$n p" "$FITNESS" | cut -f1 | tr '\n' ' ')
		if [[ $(echo -n $ROW | wc -m) -gt 0 ]]; then
			echo -en "$ROW\n" >> $IND_MATRIX
			#store maximum row length
			CURR_ROW_LEN=$(($n - $PREV_EVAL + 1))
			if [[ $CURR_ROW_LEN -gt $LONGEST_ROW ]]; then
				LONGEST_ROW=$CURR_ROW_LEN
			fi
		fi
		#update start row
		PREV_EVAL=$(($n+1))			
	done
done < "$STATS"
#echo $LONGEST_ROW

#compose image matrix
ROWS=0
while read line; do
	imgs=$(echo $line | sed 's/\>/.png/g')
		
	skip=0 #FALSE
	for f in $imgs; do
		if [[ ! -f "$f" ]]; then
  			skip=1
  			echo "Skipping row ($f not found)"
  			break
  		fi
  	done

	if [[ $skip -eq 0 ]]; then
		ROWS=$(($ROWS+1))
		convert $imgs +append "row_$ROWS.png"
		#echo file{$line}.txt		
	fi
done < $IND_MATRIX


if [[ $ROWS -gt 0 ]]; then
	rows=$(seq 1 $ROWS | sed 's/^/row_/g; s/$/.png/g')
	echo $rows
	convert $rows -append composite.png
	#rm -rf $rows
else
	echo "0 images found"
fi

