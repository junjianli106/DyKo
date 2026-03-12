#!/bin/bash

cd ../..

gpuid=3
fold=3

modelnames=(
#        'AMIL/AMIL.yaml'
#        'TransMIL/TransMIL.yaml'
#        'ILRA/ILRA.yaml'
#        'DeepGraphConv/DeepGraphConv.yaml'
#        'PatchGCN/PatchGCN.yaml'
#        'RRTMIL/RRTMIL.yaml'
#        'WiKG/WiKG.yaml'
#        'DSMIL/DSMIL.yaml'
#        'ZoomMIL/ZoomMIL.yaml'
##        'CoOpMIL/CoOpMIL.yaml'
#        'TopMIL/TopMIL.yaml'
#        'ViLaMIL/ViLaMIL.yaml'
#        'FOCUS/FOCUS.yaml'
      'DyKo/DyKo.yaml'
      )

datasetnames=("RCC" "CAMELYON16" "NSCLC"  "UBC") #

##############################################
N_SHOTS=(4 8 16)

for datasetname in "${datasetnames[@]}"
do
  for modelname in "${modelnames[@]}"
    do
      for N_SHOT in "${N_SHOTS[@]}"
      do
        config="$datasetname/$modelname"
        echo ' '
        echo ' '

        # 判断是否是 BRACS-3 或 BRACS-7
        if [[ "$datasetname" == "BRACS-3" || "$datasetname" == "BRACS-7" ]]; then
          python train.py --stage='train' --config="Cls/$config" --gpus=$gpuid --fold=$fold --task='cls' --seed=$((2025 + fold))  --n_shot=$N_SHOT
          python train.py --stage='test' --config="Cls/$config" --gpus=$gpuid --fold=$fold --task='cls' --seed=2025  --n_shot=$N_SHOT
        elif [ "$datasetname" == "CAMELYON16"  ]; then
          python train.py --stage='train' --config="Cls/$config" --gpus=$gpuid --fold=$fold --task='cls' --seed=1024 --n_shot=$N_SHOT
          python train.py --stage='test' --config="Cls/$config" --gpus=$gpuid --fold=$fold --task='cls' --seed=1024  --n_shot=$N_SHOT
        else
          python train.py --stage='train' --config="Cls/$config" --gpus=$gpuid --fold=$fold --task='cls' --seed=2025  --n_shot=$N_SHOT
          python train.py --stage='test' --config="Cls/$config" --gpus=$gpuid --fold=$fold --task='cls' --seed=2025  --n_shot=$N_SHOT
        fi
      done
    done
done
# python train.py --stage='train' --config="Cls/BRCA/TopMIL/TopMIL.yaml" --gpus=0 --fold=0 --task='cls' --seed=2025  --n_shot=16