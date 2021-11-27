mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
log_name="LOG_"$now"_$1"
gpu=${2:-0}
other_options=$3
CUDA_VISIBLE_DEVICES=$gpu python -u train.py --name $1 --log_name $log_name $other_options 2>&1|tee log/$log_name.log