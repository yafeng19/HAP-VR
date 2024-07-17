

python evaluation.py --dataset FIVR-5K \
    --dataset_hdf5 ../video_data/FIVR_5K/features/fivr_5k.hdf5 \
    --model_path ../experiments/model.pth

python evaluation.py --dataset FIVR-200K \
    --dataset_hdf5 ../video_data/FIVR_200K/features/fivr_200k.hdf5 \
    --model_path ../experiments/model.pth

python evaluation.py --dataset EVVE \
    --dataset_hdf5 ../video_data/EVVE/features/evve.hdf5 \
    --model_path ../experiments/model.pth

python evaluation.py --dataset SVD \
    --dataset_hdf5 ../video_data/SVD/svd.hdf5 \
    --model_path ../experiments/model.pth