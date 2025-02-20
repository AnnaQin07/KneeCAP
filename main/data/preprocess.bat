@echo off


set dataset_root=%1

python preprocess.py --img_dir %dataset_root%\images ^
                     --train_mask_dir %dataset_root%\masks ^
                     --test_mask_dir %dataset_root%\mask_test ^
                     --train_save_dir %dataset_root%\split\train ^
                     --test_save_dir %dataset_root%\split\test ^
                     --with_mask %2 ^
                     --sdm_version %3