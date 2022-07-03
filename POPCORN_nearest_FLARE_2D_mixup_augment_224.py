from training import  train_on_labeled_only2D, POPCORN2D
"""
train_on_labeled_only2D(regularized=True,save=False,  loss_weights=[1,0.01],Epoch=1000,datafolder='dataset/train_aug1_/',datafolder_val='dataset/val_aug1_/',
                        batch_size=64 ,early_stop_treshold=30, da_type="mixup",k_list=[2], axis=2,
                        modality='T1_FLAIR',img_size=[224,224],Snapshot=True)

train_on_labeled_only2D(regularized=True,save=False,  loss_weights=[1,0.01],Epoch=1000,datafolder='dataset/train_aug0_/',datafolder_val='dataset/val_aug0_/',
                        batch_size=64 ,early_stop_treshold=30, da_type="mixup",k_list=[1], axis=1,
                        modality='T1_FLAIR',img_size=[224,224],Snapshot=True)

train_on_labeled_only2D(regularized=True,save=False,  loss_weights=[1,0.01],Epoch=1000,datafolder='dataset/train_aug2_/',datafolder_val='dataset/val_aug2_/',
                        batch_size=64 ,early_stop_treshold=30, da_type="mixup",k_list=[0], axis=0,
                        modality='T1_FLAIR',img_size=[224,224],Snapshot=True)
"""
POPCORN2D(img_size=[128,128], dataset_path="/lib/", Epoch_per_step=5, increment=50, datafolder='data_nearest2D',dataselection_strategy='nearest',batch_size=64,
                resume=False, resume_after_adding_pseudo_of_step=1, load_precomputed_features=False, unlabeled_dataset="FLARE",recompute_distance_each_step=False,Snapshot=True, da_type="mixup",
                  regularized=True, loss_weights=[1,0.01], k=5, in_filepath=["weights/SUPERVISED_2D_T1_FLAIR_regularized224_mixupresnet18dice_axis_0_k0_.pt",  
                                                                              "weights/SUPERVISED_2D_T1_FLAIR_regularized224_mixupresnet18dice_axis_1_k1_.pt",
                                                                                "weights/SUPERVISED_2D_T1_FLAIR_regularized224_mixupresnet18dice_axis_2_k2_.pt"])
