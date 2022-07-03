from training import train_on_labeled_only_Flare

train_on_labeled_only_Flare(regularized=True,save=True,  loss_weights=[1,0.01],Epoch=500,
                        batch_size=1 ,early_stop_treshold=30, da_type="mixup",
                         nbNN=[3,3,3],ps=[128,128,128],Snapshot=True)