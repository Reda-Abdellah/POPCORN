import os, glob, sys
import numpy as np
import nibabel as nii
from keras.models import load_model
import modelos
from utils import *
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
import losses
from keras import backend as K

#original version tensorflow
def POPCORN_tensorflow( ps=[64,64,64], dataset_path="/lib/", Epoch_per_step=2, increment_new_data=200, datafolder='data_nearest/',dataselection_strategy='nearest',
                resume=False, resume_after_adding_pseudo_of_step=1, load_precomputed_features=False, unlabeled_dataset="volbrain",recompute_distance_each_step=True,
                  regularized=True, loss_weights=[1,0.01], k=5):

    
    if(regularized):
        model = modelos.load_UNET3D_bottleneck_regularized(ps[0],ps[1],ps[2],1,2,24,0.5,groups=8)
        model.compile(optimizer=optimizers.Adam(0.0001), loss=[losses.mdice_loss,losses.BottleneckRegularized],loss_weights=loss_weights)
        fun = K.function([model.input, K.learning_phase()],[model.output[0]])
        in_filepath='weights/SUPERVISED_regularized.h5'
    else:
        model=modelos.load_UNET3D_SLANT27_v2_groupNorm(ps[0],ps[1],ps[2],1,2,24,0.5)
        model.compile(optimizer=optimizers.Adam(0.0001), loss=losses.mdice_loss, metrics=[utils.mdice])
        fun = get_bottleneck_features_func(model)
        in_filepath='weights/SUPERVISED_noreg.h5'

    if(unlabeled_dataset=="volbrain"):
        listaFLAIR = sorted(glob.glob(dataset_path+"/volbrain_qc/n_mfmni*flair*.nii*"))
        listaMASK = sorted(glob.glob(dataset_path+"/volbrain_qc/mask*.nii*"))
        listaMASK = np.array(listaMASK)
    elif(unlabeled_dataset=="isbi_test"):
        listaFLAIR = sorted(glob.glob(dataset_path+"/ISBI_preprocess/test*flair*.nii*"))

    listaFLAIR=np.array(listaFLAIR)

    lib_path_3 = os.path.join(dataset_path,"isbi_final_train_preprocessed")

    listaFLAIR_3=keyword_toList(path=lib_path_3,keyword="flair")
    listaSEG1_3=keyword_toList(path=lib_path_3,keyword="mask1")
    listaSEG2_3=keyword_toList(path=lib_path_3,keyword="mask2")

    listaFLAIR_labeled= np.array(listaFLAIR_3)

    unlabeled_indxs= range(len(listaFLAIR))
    pseudolabeled_indxs=[]
    unlabeled_num=len(unlabeled_indxs)
    pseudolabeled_num=len(pseudolabeled_indxs)
    labeled_num=len(listaFLAIR_labeled)

    update_labeled_folder_flair(listaFLAIR_3,listaSEG1_3,listaMASK=None,datafolder=datafolder,numbernotnullpatch=15)
    update_labeled_folder_flair(listaFLAIR_3,listaSEG2_3,listaMASK=None,datafolder=datafolder,numbernotnullpatch=15)


    step=0
    model.load_weights(in_filepath)

    if( (not recompute_distance_each_step) and dataselection_strategy=='nearest'):
        if(load_precomputed_features):
            print('loading precomputed features...')
            rank_distance=np.load('rank_distance_volbrain_tsne3_regularized.npy')
        else:
            print('computing bottleneck_features...')
            bottleneck_features_labeled,file_names= features_from_names_flair(listaFLAIR_labeled,fun)
            bottleneck_features_unlabeled,file_names= features_from_names_flair(listaFLAIR[unlabeled_indxs],fun,listaMASK[unlabeled_indxs])
            print('computing projection to simplar dataplane...')
            rank_distance= tsne_rank(bottleneck_features_unlabeled,bottleneck_features_labeled,n_components=3)
            np.save('rank_distance_volbrain_tsne3_regularized.npy',rank_distance)

    if(resume):
        for it in range(resume_after_adding_pseudo_of_step):
            print('resuming training...')
            if(dataselection_strategy=='nearest'):
                new_pos_in_features = give_dist_for_Kclosest(rank_distance,n_indxs=increment_new_data,k=k)
            elif(dataselection_strategy=='random'):
                np.random.seed(43+it+1)
                new_pseudo = np.array(unlabeled_indxs)
                np.random.shuffle(new_pseudo)
                new_pseudo =new_pseudo[:increment_new_data]
                new_pseudo=new_pseudo.tolist()
                new_pos_in_features= new_pseudo
            not_new_pos_in_features = [x for x in range(unlabeled_num) if x not in new_pos_in_features]
            #update indexes
            pseudolabeled_indxs= pseudolabeled_indxs+ new_pos_in_features
            unlabeled_indxs =  [x for x in unlabeled_indxs if x not in pseudolabeled_indxs]
            #update num
            unlabeled_num=len(unlabeled_indxs)
        step=resume_after_adding_pseudo_of_step-1
        if(not step==0):
            model.load_weights(out_filepath(step))


    #Training
    while(unlabeled_num>increment_new_data):
        step=step+1
        out_filepath= 'weights/'+sys.argv[0].replace('.py','_step')+"%02d" % (x)+'.h5'
        print('step: '+str(step))
        print('loading new data...')
        if( resume and step==resume_after_adding_pseudo_of_step):
            print('resuming..')
        else:
            
            if(dataselection_strategy=='nearest'):
                if(recompute_distance_each_step):
                    bottleneck_features_labeled,file_names= features_from_names_flair(listaFLAIR_labeled,fun)
                    bottleneck_features_unlabeled,file_names= features_from_names_flair(listaFLAIR[unlabeled_indxs],fun,listaMASK[unlabeled_indxs])
                    if(len(pseudolabeled_indxs)>0):
                        bottleneck_features_pseudolabeled,file_names= features_from_names_flair(listaFLAIR[pseudolabeled_indxs],fun,listaMASK[pseudolabeled_indxs])
                        bottleneck_features_labeled=  np.concatenate((bottleneck_features_labeled,bottleneck_features_pseudolabeled),axis=0)
                    print('computing projection to simplar dataplane...')
                    rank_distance= tsne_rank(bottleneck_features_unlabeled,bottleneck_features_labeled,n_components=3)
                    np.save('step'+str(step)+'.npy',rank_distance)
                new_pos_in_features = give_dist_for_Kclosest(rank_distance,n_indxs=increment_new_data,k=k)
            elif(dataselection_strategy=='random'):
                np.random.seed(43+it+1)
                new_pseudo = np.array(unlabeled_indxs)
                np.random.shuffle(new_pseudo)
                new_pseudo =new_pseudo[:increment_new_data]
                new_pseudo=new_pseudo.tolist()
                new_pos_in_features= new_pseudo
            print(new_pos_in_features)
            not_new_pos_in_features = [x for x in range(unlabeled_num) if x not in new_pos_in_features]
            pseudolabeled_indxs= pseudolabeled_indxs+ new_pos_in_features
            unlabeled_indxs =  [x for x in unlabeled_indxs if x not in pseudolabeled_indxs]
            #update num
            unlabeled_num=len(unlabeled_indxs)
            update_data_folder_flair(model,new_pos_in_features,listaFLAIR,listaMASK,datafolder=datafolder,regularized=regularized)
        train_files_bytiles=[]
        for i in range(27):
            train_files_bytiles.append(keyword_toList(datafolder,"x*tile_"+str(i)+".npy") )


        print('training with new data...')
        
        numb_data= len(sorted(glob.glob(datafolder+"x*.npy")))
        if(regularized):
            result=model.fit_generator(data_gen_iqda_2it(datafolder,train_files_bytiles,sim='output_diff'), 
                steps_per_epoch=numb_data,
                epochs=Epoch_per_step)
        else:
            result=model.fit_generator(data_gen_iqda(datafolder),
                steps_per_epoch=numb_data,
                epochs=Epoch_per_step)

        model.save_weights(out_filepath(step))


def train_on_labeled_only(regularized=True):
    nbNN=[3,3,3]
    ps=[64,64,64]


    datafolder='data64flair/'
    datafolder_val='data64flair_val/'
    Epoch=100
    lib_path_1 = os.path.join("..","lib","MS_O")
    lib_path_2 = os.path.join("..","lib","msseg")
    lib_path_3 = os.path.join("..","lib","isbi_final_train_preprocessed")


    listaT1_2=keyword_toList(path=lib_path_2,keyword="t1")
    listaFLAIR_2=keyword_toList(path=lib_path_2,keyword="flair")
    listaSEG_2=keyword_toList(path=lib_path_2,keyword="mask1")
    listaT1_3=keyword_toList(path=lib_path_3,keyword="mprage")
    listaFLAIR_3=keyword_toList(path=lib_path_3,keyword="flair")
    listaSEG1_3=keyword_toList(path=lib_path_3,keyword="mask1")
    listaSEG2_3=keyword_toList(path=lib_path_3,keyword="mask2")

    update_labeled_folder(listaT1_2,listaFLAIR_2,listaSEG_2,listaMASK=None,datafolder=datafolder_val,nbNN=nbNN,ps=ps,numbernotnullpatch=10)
    update_labeled_folder(listaT1_3,listaFLAIR_3,listaSEG1_3,listaMASK=None,datafolder=datafolder,nbNN=nbNN,ps=ps,numbernotnullpatch=15)
    update_labeled_folder(listaT1_3,listaFLAIR_3,listaSEG2_3,listaMASK=None,datafolder=datafolder,nbNN=nbNN,ps=ps,numbernotnullpatch=15)

    t0=time.time()


    # load model (UNET3D)
    nf = 24
    nc = 2
    ch = 1
    drop=0.5

    model=modelos.load_UNET3D_bottleneck_regularized(ps[0],ps[1],ps[2],ch,nc,nf,drop,groups=8)

    model.compile(optimizer=optimizers.Adam(0.0001), loss=[losses.mdice_loss,losses.BottleneckRegularized],loss_weights=[1,0.01])
    model.summary()
    filepath="One_2mods_2it02same_loss3_1_001_64_flair_only_"+str(ps[0])+"_ISBI_gen_IQDA_.h5"
    #model.load_weights(filepath)

    savemodel=ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)
    #savemodel_stop=EarlyStopping(monitor='val_mdice', patience=5, verbose=1, mode='max', baseline=None, restore_best_weights=False)

    train_files_bytiles=[]
    val_files_bytiles=[]
    for i in range(27):
        train_files_bytiles.append(keyword_toList(datafolder,"x*tile_"+str(i)+".npy") )
        val_files_bytiles.append(keyword_toList(datafolder_val,"x*tile_"+str(i)+".npy"))

    #test_inter_sim(train_files_bytiles)

    numb_data= len(sorted(glob.glob(datafolder+"x*.npy")))
    numb_data_val= len(sorted(glob.glob(datafolder_val+"x*.npy")))

    if(regularized):

        result=model.fit_generator(data_gen_iqda_2it(datafolder,train_files_bytiles,sim='segmentation_distance'),#data_gen(),
                    steps_per_epoch=numb_data,
                    validation_data=data_gen_iqda_2it(datafolder_val,val_files_bytiles,sim='segmentation_distance'),
                    validation_steps=numb_data_val/27,callbacks=[savemodel],
                    epochs=Epoch)

    else:
        result=model.fit_generator(data_gen_iqda(datafolder=datafolder),
                    steps_per_epoch=numb_data,
                    validation_data=data_gen_iqda(datafolder=datafolder_val),
                    validation_steps=numb_data_val/27,callbacks=[savemodel],
                    epochs=Epoch)
                    
    model.reset_states()

    K.clear_session()
    gc.collect() #free memory
    os.chdir(Rootpath)

    t1=time.time()
    print("Processing time=",t1-t0)
    trainingTime = np.array([t1-t0])
    np.savetxt('Training_time_1mm.txt', trainingTime, fmt="%5.2f")
