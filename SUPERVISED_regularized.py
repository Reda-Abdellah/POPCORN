from training import train_on_labeled_only

train_on_labeled_only(regularized=True,save=False,  loss_weights=[1,0.01],Epoch=500,
                        batch_size=4 ,early_stop_treshold=30, da_type="iqda_v2",
                        modality='T1_FLAIR', nbNN=[5,5,5],ps=[64,64,64],Snapshot=True)