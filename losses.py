def mdice_loss_pytorch(y_pred, y_true):
    acu=0
    n_class=y_true.size(1)
    #epsilon=1
    epsilon=0.01
    for i in range(0,n_class):
        b=y_true[:,i,:,:,:]
        a=y_pred[:,i,:,:,:]
        y_int = a[:]*b[:]
        vol_pred = a[:].sum()
        vol_gt = b[:].sum()
        vol_int = y_int.sum()
        #print(vol_int)
        acu=acu+ (2*vol_int+ epsilon) / (vol_gt +vol_pred + epsilon)
    acu=acu/n_class
    return 1-acu
