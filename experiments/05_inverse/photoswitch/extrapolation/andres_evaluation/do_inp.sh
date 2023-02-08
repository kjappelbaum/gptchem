#!/bin/bash -l

for folder in 6 7 8 9 10 11; do
  # print folder name and change directory
  echo $folder
  cd $folder
  for i in *.xyz; do
    echo "${i%.*}"
    mkdir -p ${i%.*}
    cd ${i%.*}
    mkdir -p tddft
    cd ..
    tail -n +3 $i >temp
    cat ../temp_opt.com temp >input.com
    echo " " >>input.com
    mv input.com "${i%.*}"/
    cp ../temp_tddft.com "${i%.*}"/tddft/input.com
    cp ../temp_run.sh "${i%.*}"/tddft/run.sh
    cp ../temp_run.sh "${i%.*}"/run.sh
    rm temp
  done
  cd ..
done
