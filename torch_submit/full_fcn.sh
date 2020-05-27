#!/bin/bash
cd ${WORKING_PATH}
export -n JAVA_TOOL_OPTIONS="-Xms2000m -Xmx10000m"
export PATH=${WORKING_PATH}/bin:$PATH
hadoop fs -get hdfs://hobot-bigdata/user/liangchen.song/data/lib.tar
tar xf lib.tar 

hdfs dfs -get /user/liangchen.song/data/seg/lmdb/cityscapes_train ./data 
hdfs dfs -get /user/liangchen.song/data/seg/lmdb/cityscapes_val ./data 
hdfs dfs -get /user/liangchen.song/data/seg/lmdb/gta5_trans_valid ./data/
export LD_LIBRARY_PATH="lib:${LD_LIBRARY_PATH}"
env
CMD=`pwd`
echo ${CMD}
export PYTHONPATH=${CMD}:$PYTHONPATH
export PYTHONPATH=/cluster_home/libs:$PYTHONPATH
export CLASSPATH=$HADOOP_PREFIX/lib/classpath_hdfs.jar
export JAVA_TOOL_OPTIONS="-Xms2000m -Xmx10000m"

# ./bin/python  train_seg_fcn.py --batch_size 1 --data_dir ./data/gta5_trans_valid --data_list /cluster_home/dataset/seg/gta5/valid_imagelist.txt --data_dir_target ./data/cityscapes_val --data_list_target /cluster_home/dataset/seg/cityscapes/val.txt --model_path_prefix /cluster_home/models/ --input_size 1024,512 --learning_rate 2e-10 --momentum 0.99 --num_epoch 4 --print_freq 100 --tensorboard_log_dir /job_tboard/

# ./bin/python  train_gen_seg.py --batch_size 1 --data_dir ./data/gta5_trans_valid --data_list /cluster_home/dataset/seg/gta5/valid_imagelist.txt --data_dir_target ./data/cityscapes_train --data_list_target /cluster_home/dataset/seg/cityscapes/train.txt --data_dir_val ./data/cityscapes_val --data_list_val /cluster_home/dataset/seg/cityscapes/val.txt --model_path_prefix /cluster_home/models/ --resume ./data/out --fcn_name final_fcn.pth --input_size 1024,512 --learning_rate 1e-4 --D_learning_rate 1e-4 --momentum 0.9 --warm_up_epoch 2 --num_epoch 3 --print_freq 100 --show_img_freq 100 --checkpoint_freq 10000 --optimizer adam --lambda_values 1,0,0,1e-2 --tensorboard_log_dir /job_tboard/

# ./bin/python  fine_tune_seg.py --batch_size 1 --data_dir_target ./data/cityscapes_train --data_list_target /cluster_home/dataset/seg/cityscapes/train.txt --data_dir_val ./data/cityscapes_val --data_list_val /cluster_home/dataset/seg/cityscapes/val.txt --model_path_prefix /cluster_home/models/ --resume ./data/out --gen_name gen.pth --input_size 1024,512 --learning_rate 1e-4 --momentum 0.99 --num_epoch 50 --print_freq 100 --show_img_freq 100 --percent 0.8 --tensorboard_log_dir /job_tboard/ --fcn_name final_fcn.pth

# load step1

cp /cluster_home/fcn-full-gen-20181229-102357/output/final_fcn.pth ./data/out

for (( c=1; c<=10; c++ ))
do
    mkdir ./data/out_$c
    mv ./data/out/final_fcn.pth ./data/out_$c

    ./bin/python  train_gen_seg.py --batch_size 1 --data_dir ./data/gta5_trans_valid --data_list /cluster_home/dataset/seg/gta5/valid_imagelist.txt --data_dir_target ./data/cityscapes_train --data_list_target /cluster_home/dataset/seg/cityscapes/train.txt --data_dir_val ./data/cityscapes_val --data_list_val /cluster_home/dataset/seg/cityscapes/val.txt --model_path_prefix /cluster_home/models/ --resume ./data/out_$c --fcn_name final_fcn.pth --input_size 1024,512 --learning_rate 1e-4 --D_learning_rate 1e-4 --momentum 0.9 --warm_up_epoch 1 --num_epoch 5 --print_freq 100 --show_img_freq 100 --checkpoint_freq 200 --optimizer adam --lambda_values 1,0,0,5e-3 --tensorboard_log_dir /job_tboard/ --save_path_prefix ./data/out_$c

    ./bin/python  fine_tune_seg.py --batch_size 1 --data_dir_target ./data/cityscapes_train --data_list_target /cluster_home/dataset/seg/cityscapes/train.txt --data_dir_val ./data/cityscapes_val --data_list_val /cluster_home/dataset/seg/cityscapes/val.txt --model_path_prefix /cluster_home/models/ --resume ./data/out_$c --gen_name gen.pth --input_size 1024,512 --learning_rate 1e-4 --momentum 0.99 --num_epoch 3 --print_freq 100 --show_img_freq 100 --percent 0.8 --tensorboard_log_dir /job_tboard/ --fcn_name final_fcn.pth --checkpoint_freq 500 --save_path_prefix ./data/out_$c

    mv ./data/out_$c/final_fcn.pth ./data/out/ 
    mv ./data/out_$c /job_data/
done

