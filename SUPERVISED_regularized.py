from training import train_on_labeled_only

os.environ["CUDA_VISIBLE_DEVICES"]='0'

train_on_labeled_only(True)