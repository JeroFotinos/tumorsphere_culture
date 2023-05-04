#!/bin/sh


for realiz in {1..10}
do

for rs in  0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.72 0.74 0.76 0.78
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

