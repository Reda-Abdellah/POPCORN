import torch 

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

def mdice_loss_pytorch2D(y_pred, y_true):
    acu=0
    n_class=y_true.size(1)
    epsilon=1
    #epsilon=0.01
    for i in range(0,n_class):
        b=y_true[:,i,:,:]
        a=y_pred[:,i,:,:]
        y_int = a[:]*b[:]
        vol_pred = a[:].sum()
        vol_gt = b[:].sum()
        vol_int = y_int.sum()
        #print(vol_int)
        acu=acu+ (2*vol_int) / (vol_gt +vol_pred + epsilon)
    acu=acu/n_class
    return 1-acu

def CE(x,y):
    smooth= 1e-6
    return (-y*torch.log(x+smooth)).mean()


class DC_and_CE_loss_custom2D():
    def __init__(self, weight_ce=1, weight_dice=1):
            self.weight_dice = weight_dice
            self.weight_ce = weight_ce
            
    def loss(self, net_output, target):
        dc_loss = mdice_loss_pytorch2D(net_output, target) if self.weight_dice != 0 else 0
        ce_loss = CE( net_output, target) if self.weight_ce != 0 else 0
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result