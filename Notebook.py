# import 
import warnings
warnings.filterwarnings('ignore')

import os.path
from os import path
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches
from sklearn import tree
from datetime import datetime
import pytz
import ast
from tqdm import tqdm
import copy
import pickle 
import os

# Deep Learning
#import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio 

from sklearn.metrics import cohen_kappa_score,f1_score,accuracy_score, precision_score, recall_score, classification_report, roc_auc_score, \
    hamming_loss, confusion_matrix
from scipy.special import softmax
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import xgboost as xgb

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn.preprocessing import MultiLabelBinarizer

pd.set_option('display.max_columns', None)
plt.rcParams.update({'font.size': 22})


###################################################
# Functions
def get_splits(stream_subset, with_user = True):
    stream_subset_filtered = stream_subset[["user_id","media_id",'readable_time', 'weekday', 'user_age', 
                                            'gender_num','device_num','location_num','network_num', 'x_day', 
                                             'y_day', 'x_time', 'y_time', 'matches']]
    if (not with_user):    
        stream_subset_filtered = stream_subset_filtered.drop(['user_age', 'gender_num','location_num'],axis=1)

    X = stream_subset_filtered.iloc[:,:-1]
    y = stream_subset_filtered.iloc[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1, stratify=y)

    X_train = X_train.astype({"weekday": int}); X_train = X_train.astype({"readable_time": int})
    X_test = X_test.astype({"weekday": int}); X_test = X_test.astype({"readable_time": int})
    
    return X_train, X_test, y_train, y_test

def accuracy_at_k(y_pred_prob, y_test, clf, k = 3):
    predictions_array = np.zeros([len(y_test),k],dtype=int)
    le = preprocessing.LabelEncoder()
    le.classes_ = clf.classes_
    acc_k = 0; weighted_acc_k = 0
    
    for counter in range(len(y_test)):
        predictions_array[counter] = np.flip(np.argsort(y_pred_prob[counter])[-k:])
    
    for n in range(k):
        nth_preds = predictions_array[:,n]
        nth_preds_labels = le.inverse_transform(nth_preds)
        nth_acc = accuracy_score(y_test, nth_preds_labels)
        acc_k += nth_acc 
        weighted_acc_k += (nth_acc/(n+1))
    return acc_k, weighted_acc_k

# Evaluation scripts
def evaluate_model(test_pred_prob, test_classes):
    lb = preprocessing.LabelBinarizer()
    lb.fit(test_classes)
    test_pred = lb.transform(np.argmax(test_pred_prob, axis=-1))
    # Accuracy
    accuracy = 100 * accuracy_score(test_classes, test_pred)
    print("Exact match accuracy is: " + str(accuracy) + "%")
    # Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    auc_roc = roc_auc_score(test_classes, test_pred_prob)
    print("Macro Area Under the Curve (AUC) is: " + str(auc_roc))
    auc_roc_micro = roc_auc_score(test_classes, test_pred_prob, average="micro")
    print("Micro Area Under the Curve (AUC) is: " + str(auc_roc_micro))
    auc_roc_weighted = roc_auc_score(test_classes, test_pred_prob, average="weighted")
    print("Weighted Area Under the Curve (AUC) is: " + str(auc_roc_weighted))
    return accuracy, auc_roc

def create_analysis_report(model_output, groundtruth, LABELS_LIST):
    # Create a dataframe where we keep all the evaluations, starting by prediction accuracy
    lb = preprocessing.LabelBinarizer()
    lb.fit(groundtruth)
    model_output_rounded = lb.transform(np.argmax(model_output, axis=-1))
    
    accuracies_perclass = sum(model_output_rounded == groundtruth) / len(groundtruth)
    results_df = pd.DataFrame(columns=LABELS_LIST)
    results_df.index.astype(str, copy=False)
    percentage_of_positives_perclass = sum(groundtruth) / len(groundtruth)
    results_df.loc[0] = percentage_of_positives_perclass
    results_df.loc[1] = accuracies_perclass
    results_df.index = ['Ratio of positive samples', 'Model accuracy']
    
    """
    # plot the accuracies per class
    results_df.T.plot.bar(figsize=(22, 12), fontsize=18)
    plt.title('Model accuracy vs the ratio of positive samples per class')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(output_path, "accuracies_vs_positiveRate.pdf"), format="pdf")
    plt.savefig(os.path.join(output_path, "accuracies_vs_positiveRate.png"))
    """
    
    # Getting the true positive rate perclass
    true_positives_ratio_perclass = sum((model_output_rounded == groundtruth) * (groundtruth == 1)) / sum(groundtruth)
    results_df.loc[2] = true_positives_ratio_perclass
    # Get true negative ratio
    true_negative_ratio_perclass = sum((model_output_rounded == groundtruth)
                                       * (groundtruth == 0)) / (len(groundtruth) - sum(groundtruth))
    results_df.loc[3] = true_negative_ratio_perclass
    # compute additional metrics (AUC,f1,recall,precision)
    auc_roc_per_label = roc_auc_score(groundtruth, model_output, average=None)
    precision_perlabel = precision_score(groundtruth, model_output_rounded, average=None)
    recall_perlabel = recall_score(groundtruth, model_output_rounded, average=None)
    f1_perlabel = f1_score(groundtruth, model_output_rounded, average=None)
    kappa_perlabel = [cohen_kappa_score(groundtruth[:, x], model_output_rounded[:, x]) for x in range(len(LABELS_LIST))]
    results_df = results_df.append(
        pd.DataFrame([auc_roc_per_label,recall_perlabel, precision_perlabel, f1_perlabel, kappa_perlabel], columns=LABELS_LIST))
    results_df.index = ['Ratio of positive samples', 'Model accuracy', 'True positives ratio',
                        'True negatives ratio', "AUC", "Recall", "Precision", "f1-score", "Kappa score"]

    """
    # Creating evaluation plots
    plot_true_poisitve_vs_all_positives(model_output_rounded, groundtruth,
                                        os.path.join(output_path, 'TruePositive_vs_allPositives'), LABELS_LIST)
    plot_output_coocurances(model_output_rounded, os.path.join(output_path, 'output_coocurances'), LABELS_LIST)
    plot_false_netgatives_confusion_matrix(model_output_rounded, groundtruth,
                                           os.path.join(output_path, 'false_negative_coocurances'), LABELS_LIST)
    """
    results_df['average'] = results_df.mean(numeric_only=True, axis=1)
    #results_df.T.to_csv(os.path.join(output_path, "results_report.csv"), float_format="%.2f")
    return results_df

# Formatting filenames within directories [DONE ONCE]
def renameSplitFiles(SPLIT_NUMBERS, SPLIT_TYPES, LABELS_NUMBERS ,
                     GROUNDTRUTH_PATH = "/home/mounted/situational_playlist_generator/groundtruth/"):
    for splitNum in SPLIT_NUMBERS: 
        for splitTyp in SPLIT_TYPES:
            for labelNum in LABELS_NUMBERS:
                directory = GROUNDTRUTH_PATH + splitTyp + '/' + labelNum+ 'Classes/split_' + splitNum + '/'
                for filename in os.listdir(directory):               
                    if ("audio_trainset" in filename):
                        newname = "[split" + splitNum + "]audio_trainset[" + labelNum + "label_" + splitTyp + "].csv"
                        os.rename(os.path.join(directory, filename),os.path.join(directory, newname))

                    elif ("audio_testset" in filename):
                        newname = "[split" + splitNum + "]audio_testset[" + labelNum + "label_" + splitTyp + "].csv"
                        os.rename(os.path.join(directory, filename),os.path.join(directory, newname))

                    elif ("trainset" in filename):
                        newname = "[split" + splitNum + "]trainset[" + labelNum + "label_" + splitTyp + "].csv"
                        os.rename(os.path.join(directory, filename),os.path.join(directory, newname))

                    elif ("testset" in filename): 
                        newname = "[split" + splitNum + "]testset[" + labelNum + "label_" + splitTyp + "].csv"
                        os.rename(os.path.join(directory, filename),os.path.join(directory, newname))
                        
                        
######################################################################
## Situation Predictor
# load the splitted dataset and format it for the stream and audio models
def loadData_sitPred(labelNum, splitTyp, splitNum, GROUNDTRUTH_PATH):
    dataDir = GROUNDTRUTH_PATH + splitTyp + '/' + labelNum+ 'Classes/split_' + splitNum
    
    trainset = pd.read_csv(dataDir + "/[split" + splitNum + "]trainset[" 
                           + labelNum + "label_" + splitTyp + "].csv")
    
    testset = pd.read_csv(dataDir + "/[split" + splitNum + "]testset[" 
                          + labelNum + "label_" + splitTyp + "].csv")

    # Selecting SitPred data
    y_train = trainset.matches
    y_test = testset.matches
    X_train_streams = trainset.drop(["user_id","media_id","matches","fold"],axis=1)
    X_test_streams = testset.drop(["user_id","media_id","matches","fold"],axis=1)
    
    """
    # Save a version of the dataset readable for the audio model training 
    trainset.rename(columns={"media_id": "song_id"},inplace=True)
    testset.rename(columns={"media_id": "song_id"},inplace=True)
    audio_train_data = trainset[["user_id","song_id","matches"]] 
    audio_test_data = testset[["user_id","song_id","matches"]] 

    ## DO I NEED TO SAVE THEM?? (I think for the dataset pipeline??)
    audio_train_data.to_csv("/home/mounted/groundtruths/audio_trainset[4label_noUseroverlap]",index=False)
    audio_test_data.to_csv("/home/mounted/groundtruths/audio_testset[4label_noUseroverlap]",index=False)
    """
    
    return X_train_streams, y_train, X_test_streams, y_test

### Training and Evaluating XGBOOST

# Training
def train_xgb(X_train_streams, y_train):
    print("Training XGB model..")
    xgb_model = xgb.XGBClassifier(n_jobs=1).fit(X_train_streams, y_train)
    print('Accuracy of xgb classifier on training set: {:.2f}'
         .format(xgb_model.score(X_train_streams, y_train)))
    return xgb_model

# Evaluation 
def test_xgb(X_test_streams, y_test, xgb_model, label_list, results_path):
    print("Evaluating XGB model..")
    y_pred = xgb_model.predict(X_test_streams)
    report = classification_report(y_test, y_pred,labels = label_list, output_dict=True)
    report_df = pd.DataFrame.from_dict(report)
    # [TODO] SAVE_REPORT
    cm = confusion_matrix(y_test, y_pred, labels=label_list)
    cm_df = pd.DataFrame(cm,index=label_list, columns=label_list)
    # [TODO] SAVE_CONFUSION
    # predict top 3
    y_pred_prob = xgb_model.predict_proba(X_test_streams)
    acc_k, weighted_acc_k = accuracy_at_k(y_pred_prob, y_test, xgb_model, k=3)
    report_df[["accuracy@k"]] = acc_k
    report_df[["weighted accuracy@k"]] = weighted_acc_k
    print("accuracy for 3 top predictions: " + str(acc_k))
    print("Weighted accuracy for 3 top predictions: " + str(weighted_acc_k)) 
    
    np.save(results_path + "sit_pred_prop.npy", y_pred_prob)
    np.save(results_path + "sit_pred_test_gt.npy", y_test)
    report_df.to_csv(results_path + "sit_pred_report.csv")
    cm_df.to_csv(results_path + "sit_pred_confusion_matrix.csv")
    
    return report_df, cm_df, y_pred
    #MAKE DATAFRAME. SAVE ALL RESULTS IN CSV
    
    
####################################################################
# Auto-tagger model
class Conv_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2):
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
        #self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)
    def forward(self, x):
        #out = self.mp(self.relu(self.bn(self.conv(x))))
        out = self.mp(self.relu(self.conv(x)))
        return out

class user_autotagger(nn.Module):
    def __init__(self, NUM_CLASSES):
        super(user_autotagger, self).__init__()
        self.a_norming = nn.BatchNorm2d(1)
        self.u_norming = nn.BatchNorm1d(256)
        self.to_db = torchaudio.transforms.AmplitudeToDB()

        self.conv1 = Conv_2d(1,32)
        self.conv2 = Conv_2d(32,64)
        self.conv3 = Conv_2d(64,128)
        self.conv4 = Conv_2d(128,256)
        
        self.a_fc1 =  nn.Linear(256*6*40, 512)
        self.a_fc2 = nn.Linear(512, 256)
        self.a_fc3 = nn.Linear(256, 128)       
        
        self.u_embed1 = nn.Linear(EMBEDDINGS_DIM, 128+64)
        self.u_embed2 = nn.Linear(128+64, 128)
        
        self.merged_fc = nn.Linear(128+128, 128) # 256 -> results of concats
        self.drop = nn.Dropout(p=0.3)
        self.logits  = nn.Linear(128, NUM_CLASSES)
        
    def forward(self,audio_input, user_input):
        #Audio Branch 
        audio_db = self.to_db(audio_input)
        audio_norm = self.a_norming(audio_db)
        
        x_audio = self.conv1(audio_norm)
        x_audio = self.conv2(x_audio)
        x_audio = self.conv3(x_audio)
        x_audio = self.conv4(x_audio)

        x_audio = x_audio.view(x_audio.size(0), -1)
        x_audio = F.relu(self.a_fc1(x_audio))
        x_audio = F.relu(self.a_fc2(x_audio))
        x_audio = F.relu(self.a_fc3(x_audio))
        
        #User Branch
        #user_norm = self.u_norming(user_input)
        #user_norm = user_input ## [Need to figure out BatchNorm on 1D]
        #x_user = F.relu(self.u_embed1(user_norm))
        x_user = F.relu(self.u_embed1(user_input))
        x_user = F.relu(self.u_embed2(x_user))
        
        #Merged Branch
        x_conc = torch.cat((x_audio, x_user), 1)
        x_merged = torch.sigmoid(self.merged_fc(x_conc))
        x_merged = self.drop(x_merged)
        logits = self.logits(x_merged)
        output = F.softmax(logits,dim=1)
        return output, logits
    
# Defining dataset pipeline 
class UserAwareDataset(torch.utils.data.Dataset):
    def __init__(self, labels_csv, mels_folder, label_encoder,  device = 'cpu', transform = None):
        self.df = pd.read_csv(labels_csv)
        #self.embeds = pd.read_csv(embeds_csv)
        self.mels_folder = mels_folder
        self.transform = transform
        self.device = device
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        track_id = self.df["song_id"][index]
        label = self.label_encoder.transform([self.df["matches"][index]])
        label = torch.from_numpy(label)
        
        # Check it's returnning the expected format [NEED TO LOAD usr_idx_ordered and W out (change that)]
        user_id = self.df["user_id"][index]  
        usridx = usr_idx_ordered.index(user_id)
        user_embeddings = W[usridx,:]
        #user_embeddings = self.embeds[self.embeds.user_id == user_id].iloc[:,1:].values.flatten()[0]
        #user_embeddings = np.asarray(ast.literal_eval(user_embeddings))
        user_embeddings = user_embeddings.astype(np.float32) #.reshape(1,-1)
        
        # this is to ensure all mels have same shape (padded if missing)
        mel_spec = torch.zeros(1,96,646)
        try:
            loaded_spec = torch.from_numpy(np.load(os.path.join(self.mels_folder, str(track_id)+".npz"))['arr_0'])
        except:
            loaded_spec = torch.from_numpy(np.load(os.path.join(self.mels_folder, str(float(track_id))+".npz"))['arr_0'])
        if(loaded_spec.dim() == 2):
            loaded_spec = torch.unsqueeze(loaded_spec,0)
        if (loaded_spec.shape[1] != 96):
            loaded_spec = loaded_spec.permute(0,2,1)
        mel_spec[:, :, :loaded_spec.shape[2]] = loaded_spec
        #mel_spec = torch.unsqueeze(mel_spec, 0)
        if self.transform is not None:
            """MAKE DB HERE? could be faster"""
            mel_spec = self.transform(mel_spec) 

        return mel_spec, user_embeddings , label
    
# initiating dataloader 
def initialize_audio_data(labelNum, splitTyp, splitNum, label_encoder, GROUNDTRUTH_PATH, MELS_PATH):    
    dataDir = GROUNDTRUTH_PATH + splitTyp + '/' + labelNum+ 'Classes/split_' + splitNum
    trainDataDir = dataDir + "/[split" + splitNum + "]audio_trainset[" + labelNum + "label_" + splitTyp + "].csv"
    testDataDir = dataDir + "/[split" + splitNum + "]audio_testset[" + labelNum + "label_" + splitTyp + "].csv"
    
    train_instance = UserAwareDataset(trainDataDir, MELS_PATH, label_encoder)
    test_instance = UserAwareDataset(testDataDir, MELS_PATH, label_encoder)
    
    train_loader = torch.utils.data.DataLoader(train_instance,batch_size=32,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_instance,batch_size=32,shuffle=False)
    
    # Get weights for CE training (the ratio of each class in the dataset)
    train_GT = pd.read_csv(dataDir + "/[split" + splitNum + "]trainset[" 
                           + labelNum + "label_" + splitTyp + "].csv")
    values = label_encoder.transform(train_GT.matches)
    n_values = np.max(values) + 1
    train_GT = np.eye(n_values)[values]
    POS_WEIGHTS = len(train_GT)/train_GT.sum(axis=0)
    POS_WEIGHTS = [np.float32(x) for x in POS_WEIGHTS] # Do I still need this??
    POS_WEIGHTS = torch.FloatTensor(POS_WEIGHTS).to(device)
    
    #validation_instance = UserAwareDataset("/home/mounted/implicit_valid_set.csv",
    #                            "/home/mounted/groundtruths/user_embeds_existing.csv",
    #                           "/home/mounted/implicit_mels/")
    #valid_loader = torch.utils.data.DataLoader(validation_instance,batch_size=32,shuffle=True)

    return train_loader, test_loader, POS_WEIGHTS


# get autotagger
def get_autotagger(labelNum, device, POS_WEIGHTS):
    # Define loss and optimizer
    autotagger = user_autotagger(int(labelNum))
    criterion = nn.CrossEntropyLoss(weight=POS_WEIGHTS)
    #criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(autotagger.parameters(), lr=0.001, weight_decay=1e-4)

    #Decaying learning rate
    #decayRate = 0.98
    #optimizer_decayLR = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    autotagger.to(device)
    return autotagger, optimizer, criterion

# Training loop
def train_autotagger(autotagger, train_loader, optimizer, criterion):
    for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
        autotagger.train()
        epoch_loss = 0.0
        correct = 0
        # iterate the training set
        with tqdm(train_loader, unit="batch") as tepoch:
            for data in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")

                #audio_in, user_embeds, labels = data
                audio_in = data[0].to(device)
                user_embeds = data[1].to(device)
                labels = torch.squeeze(data[2]).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs,logits = autotagger(audio_in,user_embeds)
                loss = criterion(logits, labels) #Notice, CE in pytorch requires targets as indices
                loss.backward()
                optimizer.step()

                _, predicted_idx = torch.max(outputs.data, 1)
                correct += (predicted_idx == labels).sum().item()

                # compute epoch loss
                epoch_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

    print('Finished Training')
    
# Evaluation of the model
def test_autotagger(autotagger, test_loader, y_test, label_encoder, labels_list, results_path):
    #test_GT = pd.read_csv(GROUNDTRUTH_PATH + "trainset[" + num_class + "label_" + split_type + "].csv")
    values = label_encoder.transform(y_test)
    n_values = np.max(values) + 1
    test_labels = np.eye(n_values)[values]
    test_gt_idx = np.zeros([len(y_test),1], dtype=float)
    test_pred_prob = np.zeros_like(test_labels, dtype=float)
    #test_one_hot = np.zeros_like(test_classes, dtype=float)

    autotagger.eval()
    with torch.no_grad():
        for step, (audio_in,user_embeds,labels) in enumerate(test_loader):
            audio_in = audio_in.to(device); user_embeds = user_embeds.to(device)
            outputs, logits = autotagger(audio_in,user_embeds)
            #_, predicted_idx = torch.max(outputs.data, 1)
            #_, labels_idx = torch.max(labels, 1)

            # save all predictions and GT in on single array (redoing the GT to be aligned with preds)
            start_idx = (step * BATCH_SIZE); end_idx = (step * BATCH_SIZE) + labels.size(0)
            test_gt_idx[start_idx:end_idx, :] = labels.cpu()
            test_pred_prob[start_idx:end_idx, :] = outputs.cpu()

    """Do you wanna save the user_ids and tracks? """
    #test_one_hot = np.zeros_like(test_classes, dtype=float)
    #test_song_ids = np.zeros([test_classes.shape[0],1])
    #test_user_ids = np.zeros([test_classes.shape[0],1])

        #test_song_ids[start_idx:end_idx] = test_batch[3].reshape([-1, 1])
        #test_user_ids[start_idx:end_idx] = test_batch[4].reshape([-1, 1])

        #test_pred_classes = 

    accuracy_out, auc_roc = evaluate_model(test_pred_prob, test_labels)
    results = create_analysis_report(test_pred_prob, test_labels, labels_list)
    results[["Exact Match"]] = accuracy_out
    
    np.save(results_path + "autotagger_test_gt.npy", test_labels)
    np.save(results_path + "autotagger_pred_prob.npy", test_pred_prob)
    results.to_csv(results_path + "autotagger_report.csv")
    return results, test_pred_prob, test_labels

# Match the prediction of the test set from the two models [MAKE RESULTS PER LABEL???]
def joint_evaluation(y_pred, test_pred_prob,test_labels, label_encoder, labels_list, results_path):
    Audio_preds = label_encoder.inverse_transform(np.argmax(test_pred_prob,axis=1))
    GT_labels = label_encoder.inverse_transform(np.argmax(test_labels,axis=1))
    correct = 0
    not_correct_idx = []
    for counter in range(len(GT_labels)):
        if len({Audio_preds[counter], GT_labels[counter], y_pred[counter]}) == 1: ## WHAT WAS I DOING HERE??
            correct+=1
        else:
            not_correct_idx.append(counter)
    joint_accuracy = (correct/len(GT_labels))
    print("XGBOOST accuracy of mathing predictions: %.4f" % joint_accuracy)
    with open(results_path + "joint_evaluation.txt", "w") as f:
        f.write(str(joint_accuracy))
    return joint_accuracy


##############################################################################
########################################################
# MAIN LOOP
################################
# Initialize constants        
        
def main():
    #min_val_loss = 10**5 #just initialize with random big number 
    #epochs_no_improve = 0
    #n_epochs_stop = 10
    
    for splitNum in SPLIT_NUMBERS: 
        for splitTyp in SPLIT_TYPES:
            for labelNum in LABELS_NUMBERS: 
                # select right labels list
                labels_list = LABELS_LISTS[labelNum]
                results_path = RESULTS_PATH + splitTyp + '/' + labelNum+ 'Classes/split_' + splitNum + "/"

                #Situation Predictor
                X_train_streams, y_train, X_test_streams, y_test = loadData_sitPred(labelNum, 
                                                                                    splitTyp, splitNum, 
                                                                                    GROUNDTRUTH_PATH)
                xgb_model = train_xgb(X_train_streams, y_train)
                report, cm, sitPred_y_pred = test_xgb(X_test_streams, y_test, xgb_model,labels_list, results_path)

                #Autotagger

                label_encoder = preprocessing.LabelEncoder()
                label_encoder.fit(labels_list)



                train_loader, test_loader, POS_WEIGHTS = initialize_audio_data(labelNum, splitTyp, splitNum, 
                                                                               label_encoder, GROUNDTRUTH_PATH,
                                                                               MELS_PATH)


                autotagger, optimizer, criterion = get_autotagger(labelNum, device, POS_WEIGHTS)
                train_autotagger(autotagger, train_loader, optimizer, criterion)
                results, autotagger_y_pred_prob, test_labels = test_autotagger(autotagger, test_loader, y_test, 
                                                             label_encoder, labels_list, results_path)

                joint_results = joint_evaluation(sitPred_y_pred, autotagger_y_pred_prob,
                                                 test_labels, label_encoder, labels_list, results_path)
                
                model_name = model_save_path + splitTyp + '_' + labelNum + 'Classes_split_' + splitNum
                torch.save(autotagger.state_dict(),model_name)
                torch.cuda.empty_cache()
                print("================================================================")
                
                
if __name__ == "__main__":
    SPLIT_TYPES = ["ColdUser" , "ColdTrack", "WarmCase"]
    SPLIT_NUMBERS = ["1","2","3","4"]
    LABELS_NUMBERS = ["4","8","12"]
    
    INPUT_SHAPE = (1, 96, 646)
    EMBEDDINGS_DIM = 128
    BATCH_SIZE = 32
    LABELS_LISTS = {"4" : ['gym', 'party', 'sleep', 'work'],
                    "8" : ['dance','gym','morning','night', 'party','running','sleep', 'work'],
                    "12" : ['car', 'club', 'dance','gym','morning','night', 
                         'party','relax', 'running','sleep', 'train', 'work']}
    
    GROUNDTRUTH_PATH = "/home/mounted/situational_playlist_generator/groundtruth/"
    MELS_PATH = "/home/mounted/mel_folders/MELS_MERGED/"
    RESULTS_PATH = "/home/mounted/situational_playlist_generator/Results/"
    model_save_path = "/home/mounted/situational_playlist_generator/trained_models/autotagger_"
    
    with open("/home/mounted/situational_playlist_generator/groundtruth/user_idx_ordered.txt", "rb") as fp:
        usr_idx_ordered = pickle.load(fp)
    # Load embeddings matrix (acces with index mapped through the user id)
    W  = np.load("/home/mounted/situational_playlist_generator/groundtruth/users_matrix.npz.npy")
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))
    NUM_EPOCHS = 1
    
    main()