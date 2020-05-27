#!/bin/bash
cd ${WORKING_PATH}
export -n JAVA_TOOL_OPTIONS="-Xms2000m -Xmx10000m"
export PATH=${WORKING_PATH}/bin:$PATH
hadoop fs -get hdfs://hobot-bigdata/user/liangchen.song/data/lib.tar
tar xf lib.tar && hdfs dfs -get /user/liangchen.song/data/seg/lmdb/cityscapes_train ./data && hdfs dfs -get /user/liangchen.song/data/seg/lmdb/cityscapes_val ./data && hdfs dfs -get /user/liangchen.song/data/seg/lmdb/gta5_trans_valid ./data/
export LD_LIBRARY_PATH="lib:${LD_LIBRARY_PATH}"
env
CMD=`pwd`
echo ${CMD}
export PYTHONPATH=${CMD}:$PYTHONPATH
export PYTHONPATH=/cluster_home/libs:$PYTHONPATH
export CLASSPATH=$HADOOP_PREFIX/lib/classpath_hdfs.jar
export JAVA_TOOL_OPTIONS="-Xms2000m -Xmx10000m"
echo "train_gen_seg_deeplabv3.py --batch_size 1 --data_dir ./data/gta5_trans_valid --data_list /cluster_home/dataset/seg/gta5/valid_imagelist.txt --data_dir_target ./data/cityscapes_train --data_list_target /cluster_home/dataset/seg/cityscapes/train.txt --data_dir_val ./data/cityscapes_val --data_list_val /cluster_home/dataset/seg/cityscapes/val.txt --model_path_prefix /cluster_home/models/ --resume /cluster_home/models/trained/ --deeplabv3_name 30000_322.pth --input_size 1024,512 --learning_rate 1e-4 --D_learning_rate 1e-4 --momentum 0.9 --warm_up_epoch 2 --num_epoch 5 --print_freq 100 --show_img_freq 100 --checkpoint_freq 300 --optimizer adam --lambda_values 1,0,0,1e-2 --tensorboard_log_dir /job_tboard/"
./bin/python  train_gen_seg_deeplabv3.py --batch_size 1 --data_dir ./data/gta5_trans_valid --data_list /cluster_home/dataset/seg/gta5/valid_imagelist.txt --data_dir_target ./data/cityscapes_train --data_list_target /cluster_home/dataset/seg/cityscapes/train.txt --data_dir_val ./data/cityscapes_val --data_list_val /cluster_home/dataset/seg/cityscapes/val.txt --model_path_prefix /cluster_home/models/ --resume /cluster_home/models/trained/ --deeplabv3_name 30000_322.pth --input_size 1024,512 --learning_rate 1e-4 --D_learning_rate 1e-4 --momentum 0.9 --warm_up_epoch 2 --num_epoch 5 --print_freq 100 --show_img_freq 100 --checkpoint_freq 300 --optimizer adam --lambda_values 1,0,0,1e-2 --tensorboard_log_dir /job_tboard/
mv ./data/out/* /job_data/
