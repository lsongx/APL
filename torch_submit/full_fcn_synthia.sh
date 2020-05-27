#!/bin/bash
cd ${WORKING_PATH}
export -n JAVA_TOOL_OPTIONS="-Xms2000m -Xmx10000m"
export PATH=${WORKING_PATH}/bin:$PATH
hadoop fs -get hdfs://hobot-bigdata/user/liangchen.song/data/lib.tar
tar xf lib.tar 

hdfs dfs -get /user/liangchen.song/data/seg/lmdb/cityscapes_train ./data 
hdfs dfs -get /user/liangchen.song/data/seg/lmdb/cityscapes_val ./data 
cd data
time hdfs dfs -get /user/liangchen.song/data/seg/synthia.tar 
time tar xf synthia.tar
hdfs dfs -get /user/liangchen.song/data/seg/image_list/synthia.txt 
cd ..
export LD_LIBRARY_PATH="lib:${LD_LIBRARY_PATH}"
env
CMD=`pwd`
echo ${CMD}
export PYTHONPATH=${CMD}:$PYTHONPATH
export PYTHONPATH=/cluster_home/libs:$PYTHONPATH
export CLASSPATH=$HADOOP_PREFIX/lib/classpath_hdfs.jar
export JAVA_TOOL_OPTIONS="-Xms2000m -Xmx10000m"

# ./bin/python  train_seg_fcn_synthia.py --batch_size 1 --data_dir ./data/synthia --data_list ./data/synthia.txt --data_dir_target ./data/cityscapes_val --data_list_target /cluster_home/dataset/seg/cityscapes/val.txt --model_path_prefix /cluster_home/models/ --input_size 1024,512 --learning_rate 2e-10 --momentum 0.99 --num_epoch 4 --print_freq 100 --tensorboard_log_dir /job_tboard/
cp /cluster_home/fcn-full-synthia-20190116-112216/output/out/final_fcn.pth ./data/out

./bin/python  train_gen_seg_synthia.py --batch_size 1 --data_dir ./data/synthia --data_list ./data/synthia.txt --data_dir_target ./data/cityscapes_train --data_list_target /cluster_home/dataset/seg/cityscapes/train.txt --data_dir_val ./data/cityscapes_val --data_list_val /cluster_home/dataset/seg/cityscapes/val.txt --model_path_prefix /cluster_home/models/ --resume ./data/out --fcn_name final_fcn.pth --input_size 1024,512 --learning_rate 1e-4 --D_learning_rate 1e-4 --momentum 0.9 --warm_up_epoch 2 --num_epoch 3 --print_freq 100 --show_img_freq 100 --checkpoint_freq 10000 --optimizer adam --lambda_values 1,0,0,1e-2 --tensorboard_log_dir /job_tboard/

./bin/python  fine_tune_seg_synthia.py --batch_size 1 --data_dir_target ./data/cityscapes_train --data_list_target /cluster_home/dataset/seg/cityscapes/train.txt --data_dir_val ./data/cityscapes_val --data_list_val /cluster_home/dataset/seg/cityscapes/val.txt --model_path_prefix /cluster_home/models/ --resume ./data/out --gen_name gen.pth --input_size 1024,512 --learning_rate 1e-4 --momentum 0.99 --num_epoch 50 --print_freq 100 --show_img_freq 100 --percent 0.8 --tensorboard_log_dir /job_tboard/ --fcn_name final_fcn.pth

mv ./data/out /job_data/