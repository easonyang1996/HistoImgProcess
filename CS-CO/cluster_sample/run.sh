python3 ./1_feature_extract.py -m resnet-pretrained
python3 ./2_kmeans.py
python3 ./3_sampling.py
python3 ./4_get_roi.py
python3 ./5_roi_feature_extract.py -m resnet-pretrained
python3 ./6_train_test_split.py

