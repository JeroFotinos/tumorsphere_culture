#!/bin/sh


for realiz in {1..10}
do

for rs in  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do 

nombre=S-$realiz-$rs

 sed  's/val_ps/'$rs'/g' < file1.py \
|sed  's/val_realiz/'$realiz'/g'> $nombre.py


sed  's/val_nombre/'$nombre'/g' <simple.sh > .$nombre.sh


#cp .hecho.sh ~/scratch/.$nombre.$proc.sh
#cd ~/scratch/
chmod +x .$nombre.sh
chmod +x $nombre.py

qsub ./.$nombre.sh 


#unad

sleep 1
done
done


#rm .${nombre}.py

