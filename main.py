# Import the Packages
from dataset import MyDataset
from my_model import MyModel
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix

# Load the Dataset
data_dir = 'datasets/mfsc_12cl'
data = MyDataset(data_dir, augmented=False)
data_aug = MyDataset(data_dir, augmented=True)
x_tr, y_tr, x_te, y_te = data.data
x_tr_aug, y_tr_aug, x_te_aug, y_te_aug = data_aug.data
data.info()

# Load the Noisy Dataset 
kind_noise = 'my' # You Can Choose Between ['my','google']
data_dir = 'datasets/mfsc_12cl_'+kind_noise+'_noise_te'
data_noisy = MyDataset(data_dir, augmented=False)
x_tr_n_temp, y_tr_n_temp, x_te_n_temp, y_te_n_temp = data_noisy.data
x_te_n = np.concatenate((x_tr_n_temp, x_te_n_temp))
y_te_n = np.concatenate((y_tr_n_temp, y_te_n_temp))
data_noisy.info()

# Model Definitions
CNN = MyModel(x_tr, y_tr, x_te, y_te, 'CNN', augmentation=False)
CNN_aug = MyModel(x_tr_aug, y_tr_aug, x_te_aug, y_te_aug, 'CNN', augmentation=True)
CNN_AE = MyModel(x_tr, y_tr, x_te, y_te, 'CNN_AE', augmentation=False)
CNN_AE_aug = MyModel(x_tr_aug, y_tr_aug, x_te_aug, y_te_aug, 'CNN_AE', augmentation=True)
FNN_AE = MyModel(x_tr, y_tr, x_te, y_te, 'FNN_AE', augmentation=False)
FNN_AE_aug = MyModel(x_tr_aug, y_tr_aug, x_te_aug, y_te_aug, 'FNN_AE', augmentation=True)
CNN_inc = MyModel(x_tr, y_tr, x_te, y_te, 'CNN_inc', augmentation=False)
CNN_inc_aug = MyModel(x_tr_aug, y_tr_aug, x_te_aug, y_te_aug, 'CNN_inc', augmentation=True)

# Initialization of the Models
CNN.init()
CNN_aug.init()
CNN_AE.init()
CNN_AE_aug.init()
FNN_AE.init()
FNN_AE_aug.init()
CNN_inc.init()
CNN_inc_aug.init()

# Train Models
# CNN.train_classifier(epochs=50, batch_size=128)
# CNN_aug.train_classifier(epochs=50, batch_size=128)
# CNN_AE.train_autoencoder(epochs=50, batch_size=128)
# CNN_AE.train_classifier(epochs=100, batch_size=128)
# CNN_AE_aug.train_autoencoder(epochs=50, batch_size=128)
# CNN_AE_aug.train_classifier(epochs=100, batch_size=128)
# FNN_AE.train_autoencoder(epochs=50, batch_size=128)
# FNN_AE.train_classifier(epochs=100, batch_size=128)
# FNN_AE_aug.train_autoencoder(epochs=50, batch_size=128)
# FNN_AE_aug.train_classifier(epochs=100, batch_size=128)
# CNN_inc.train_classifier(epochs=50, batch_size=128)
# CNN_inc_aug.train_classifier(epochs=50, batch_size=128)

# Load Models
CNN.load_best_model()
CNN_aug.load_best_model()
CNN_AE.load_best_model(kind_partial='AE')
CNN_AE.load_best_model(kind_partial='DFNN')
CNN_AE_aug.load_best_model(kind_partial='AE')
CNN_AE_aug.load_best_model(kind_partial='DFNN')
FNN_AE.load_best_model(kind_partial='AE')
FNN_AE.load_best_model(kind_partial='DFNN')
FNN_AE_aug.load_best_model(kind_partial='AE')
FNN_AE_aug.load_best_model(kind_partial='DFNN')
CNN_inc.load_best_model()
CNN_inc_aug.load_best_model()

#################################################
#################################################
# TEST PERFORMANCES
#################################################
#################################################

### ACCURACY
# Accuracy on Clean Dataset
_, acc_CNN_c = CNN.classifier.evaluate(x_te, y_te)
_, acc_CNN_aug_c = CNN_aug.classifier.evaluate(x_te, y_te)
_, acc_CNN_AE_c = CNN_AE.classifier.evaluate(CNN_AE.encoder.predict(x_te), y_te)
_, acc_CNN_AE_aug_c = CNN_AE_aug.classifier.evaluate(CNN_AE_aug.encoder.predict(x_te), y_te)
_, acc_FNN_AE_c = FNN_AE.classifier.evaluate(FNN_AE.encoder.predict(x_te.reshape(-1, 99*40)), y_te)
_, acc_FNN_AE_aug_c = FNN_AE_aug.classifier.evaluate(FNN_AE_aug.encoder.predict(x_te.reshape(-1, 99*40)), y_te)
_, acc_CNN_inc_c = CNN_inc.classifier.evaluate(x_te, y_te)
_, acc_CNN_inc_aug_c = CNN_inc_aug.classifier.evaluate(x_te, y_te)

print('ACCURACY ON CLEAN DATASET :')
print('CNN Test Accuracy : ', acc_CNN_c)
print('CNN_aug Test Accuracy : ', acc_CNN_aug_c)
print('CNN_AE Test Accuracy : ', acc_CNN_AE_c)
print('CNN_AE_aug Test Accuracy : ', acc_CNN_AE_aug_c)
print('FNN_AE Test Accuracy : ', acc_FNN_AE_c)
print('FNN_AE_aug Test Accuracy : ', acc_FNN_AE_aug_c)
print('CNN_inc Test Accuracy : ', acc_CNN_inc_c)
print('CNN_inc_aug Test Accuracy : ', acc_CNN_inc_aug_c)

# Accuracy on Noisy Dataset
_, acc_CNN_n = CNN.classifier.evaluate(x_te_n, y_te_n)
_, acc_CNN_aug_n = CNN_aug.classifier.evaluate(x_te_n, y_te_n)
_, acc_CNN_AE_n = CNN_AE.classifier.evaluate(CNN_AE.encoder.predict(x_te_n), y_te_n)
_, acc_CNN_AE_aug_n = CNN_AE_aug.classifier.evaluate(CNN_AE_aug.encoder.predict(x_te_n), y_te_n)
_, acc_FNN_AE_n = FNN_AE.classifier.evaluate(FNN_AE.encoder.predict(x_te_n.reshape(-1, 99*40)), y_te_n)
_, acc_FNN_AE_aug_n = FNN_AE_aug.classifier.evaluate(FNN_AE_aug.encoder.predict(x_te_n.reshape(-1, 99*40)), y_te_n)
_, acc_CNN_inc_n = CNN_inc.classifier.evaluate(x_te_n, y_te_n)
_, acc_CNN_inc_aug_n = CNN_inc_aug.classifier.evaluate(x_te_n, y_te_n)

print('ACCURACY ON NOISY DATASET :')
print('CNN Test Accuracy : ', acc_CNN_n)
print('CNN_aug Test Accuracy : ', acc_CNN_aug_n)
print('CNN_AE Test Accuracy : ', acc_CNN_AE_n)
print('CNN_AE_aug Test Accuracy : ', acc_CNN_AE_aug_n)
print('FNN_AE Test Accuracy : ', acc_FNN_AE_n)
print('FNN_AE_aug Test Accuracy : ', acc_FNN_AE_aug_n)
print('CNN_inc Test Accuracy : ', acc_CNN_inc_n)
print('CNN_inc_aug Test Accuracy : ', acc_CNN_inc_aug_n)

# Precision on Clean Dataset
pre_CNN_c = precision_score(y_te.argmax(axis=-1), CNN.classifier.predict(x_te).argmax(axis=-1), average='micro')
pre_CNN_aug_c = precision_score(y_te.argmax(axis=-1), CNN_aug.classifier.predict(x_te).argmax(axis=-1), average='micro')
pre_CNN_AE_c = precision_score(y_te.argmax(axis=-1), CNN_AE.classifier.predict(CNN_AE.encoder.predict(x_te)).argmax(axis=-1), average='micro')
pre_CNN_AE_aug_c = precision_score(y_te.argmax(axis=-1), CNN_AE_aug.classifier.predict(CNN_AE_aug.encoder.predict(x_te)).argmax(axis=-1), average='micro')
pre_FNN_AE_c = precision_score(y_te.argmax(axis=-1), FNN_AE.classifier.predict(FNN_AE.encoder.predict(x_te.reshape(-1,99*40))).argmax(axis=-1), average='micro')
pre_FNN_AE_aug_c = precision_score(y_te.argmax(axis=-1), FNN_AE_aug.classifier.predict(FNN_AE_aug.encoder.predict(x_te.reshape(-1,99*40))).argmax(axis=-1), average='micro')
pre_CNN_inc_c = precision_score(y_te.argmax(axis=-1), CNN_inc.classifier.predict(x_te).argmax(axis=-1), average='micro')
pre_CNN_inc_aug_c = precision_score(y_te.argmax(axis=-1), CNN_inc_aug.classifier.predict(x_te).argmax(axis=-1), average='micro')

print('PRECISION ON CLEAN DATASET :')
print('CNN Test Precision : ', pre_CNN_c)
print('CNN_aug Test Precision : ', pre_CNN_aug_c)
print('CNN_AE Test Precision : ', pre_CNN_AE_c)
print('CNN_AE_aug Test Precision : ', pre_CNN_AE_aug_c)
print('FNN_AE Test Precision : ', pre_FNN_AE_c)
print('FNN_AE_aug Test Precision : ', pre_FNN_AE_aug_c)
print('CNN_inc Test Precision : ', pre_CNN_inc_c)
print('CNN_inc_aug Test Precision : ', pre_CNN_inc_aug_c)

# Precision on My Noisy Dataset
pre_CNN_n = precision_score(y_te_n.argmax(axis=-1), CNN.classifier.predict(x_te_n).argmax(axis=-1), average='micro')
pre_CNN_aug_n = precision_score(y_te_n.argmax(axis=-1), CNN_aug.classifier.predict(x_te_n).argmax(axis=-1), average='micro')
pre_CNN_AE_n = precision_score(y_te_n.argmax(axis=-1), CNN_AE.classifier.predict(CNN_AE.encoder.predict(x_te_n)).argmax(axis=-1), average='micro')
pre_CNN_AE_aug_n = precision_score(y_te_n.argmax(axis=-1), CNN_AE_aug.classifier.predict(CNN_AE_aug.encoder.predict(x_te_n)).argmax(axis=-1), average='micro')
pre_FNN_AE_n = precision_score(y_te_n.argmax(axis=-1), FNN_AE.classifier.predict(FNN_AE.encoder.predict(x_te_n.reshape(-1,99*40))).argmax(axis=-1), average='micro')
pre_FNN_AE_aug_n = precision_score(y_te_n.argmax(axis=-1), FNN_AE_aug.classifier.predict(FNN_AE_aug.encoder.predict(x_te_n.reshape(-1,99*40))).argmax(axis=-1), average='micro')
pre_CNN_inc_n = precision_score(y_te_n.argmax(axis=-1), CNN_inc.classifier.predict(x_te_n).argmax(axis=-1), average='micro')
pre_CNN_inc_aug_n = precision_score(y_te_n.argmax(axis=-1), CNN_inc_aug.classifier.predict(x_te_n).argmax(axis=-1), average='micro')

print('PRECISION ON NOISY DATASET :')
print('CNN Test Precision : ', pre_CNN_n)
print('CNN_aug Test Precision : ', pre_CNN_aug_n)
print('CNN_AE Test Precision : ', pre_CNN_AE_n)
print('CNN_AE_aug Test Precision : ', pre_CNN_AE_aug_n)
print('FNN_AE Test Precision : ', pre_FNN_AE_n)
print('FNN_AE_aug Test Precision : ', pre_FNN_AE_aug_n)
print('CNN_inc Test Precision : ', pre_CNN_inc_n)
print('CNN_inc_aug Test Precision : ', pre_CNN_inc_aug_n)

# RECALL
# Recall on Clean Dataset
rec_CNN_c = recall_score(y_te.argmax(axis=-1), CNN.classifier.predict(x_te).argmax(axis=-1), average='micro')
rec_CNN_aug_c = recall_score(y_te.argmax(axis=-1), CNN_aug.classifier.predict(x_te).argmax(axis=-1), average='micro')
rec_CNN_AE_c = recall_score(y_te.argmax(axis=-1), CNN_AE.classifier.predict(CNN_AE.encoder.predict(x_te)).argmax(axis=-1), average='micro')
rec_CNN_AE_aug_c = recall_score(y_te.argmax(axis=-1), CNN_AE_aug.classifier.predict(CNN_AE_aug.encoder.predict(x_te)).argmax(axis=-1), average='micro')
rec_FNN_AE_c = recall_score(y_te.argmax(axis=-1), FNN_AE.classifier.predict(FNN_AE.encoder.predict(x_te.reshape(-1,99*40))).argmax(axis=-1), average='micro')
rec_FNN_AE_aug_c = recall_score(y_te.argmax(axis=-1), FNN_AE_aug.classifier.predict(FNN_AE_aug.encoder.predict(x_te.reshape(-1,99*40))).argmax(axis=-1), average='micro')
rec_CNN_inc_c = recall_score(y_te.argmax(axis=-1), CNN_inc.classifier.predict(x_te).argmax(axis=-1), average='micro')
rec_CNN_inc_aug_c = recall_score(y_te.argmax(axis=-1), CNN_inc_aug.classifier.predict(x_te).argmax(axis=-1), average='micro')

print('RECALL ON CLEAN DATASET :')
print('CNN Test Recall : ', rec_CNN_c)
print('CNN_aug Test Recall : ', rec_CNN_aug_c)
print('CNN_AE Test Recall : ', rec_CNN_AE_c)
print('CNN_AE_aug Test Recall : ', rec_CNN_AE_aug_c)
print('FNN_AE Test Recall : ', rec_FNN_AE_c)
print('FNN_AE_aug Test Recall : ', rec_FNN_AE_aug_c)
print('CNN_inc Test Recall : ', rec_CNN_inc_c)
print('CNN_inc_aug Test Recall : ', rec_CNN_inc_aug_c)

# Recall on My Noisy Dataset
rec_CNN_n = recall_score(y_te_n.argmax(axis=-1), CNN.classifier.predict(x_te_n).argmax(axis=-1), average='micro')
rec_CNN_aug_n = recall_score(y_te_n.argmax(axis=-1), CNN_aug.classifier.predict(x_te_n).argmax(axis=-1), average='micro')
rec_CNN_AE_n = recall_score(y_te_n.argmax(axis=-1), CNN_AE.classifier.predict(CNN_AE.encoder.predict(x_te_n)).argmax(axis=-1), average='micro')
rec_CNN_AE_aug_n = recall_score(y_te_n.argmax(axis=-1), CNN_AE_aug.classifier.predict(CNN_AE_aug.encoder.predict(x_te_n)).argmax(axis=-1), average='micro')
rec_FNN_AE_n = recall_score(y_te_n.argmax(axis=-1), FNN_AE.classifier.predict(FNN_AE.encoder.predict(x_te_n.reshape(-1,99*40))).argmax(axis=-1), average='micro')
rec_FNN_AE_aug_n = recall_score(y_te_n.argmax(axis=-1), FNN_AE_aug.classifier.predict(FNN_AE_aug.encoder.predict(x_te_n.reshape(-1,99*40))).argmax(axis=-1), average='micro')
rec_CNN_inc_n = recall_score(y_te_n.argmax(axis=-1), CNN_inc.classifier.predict(x_te_n).argmax(axis=-1), average='micro')
rec_CNN_inc_aug_n = recall_score(y_te_n.argmax(axis=-1), CNN_inc_aug.classifier.predict(x_te_n).argmax(axis=-1), average='micro')

print('RECALL ON NOISY DATASET :')
print('CNN Test Recall : ', rec_CNN_n)
print('CNN_aug Test Recall : ', rec_CNN_aug_n)
print('CNN_AE Test Recall : ', rec_CNN_AE_n)
print('CNN_AE_aug Test Recall : ', rec_CNN_AE_aug_n)
print('FNN_AE Test Recall : ', rec_FNN_AE_n)
print('FNN_AE_aug Test Recall : ', rec_FNN_AE_aug_n)
print('CNN_inc Test Recall : ', rec_CNN_inc_n)
print('CNN_inc_aug Test Recall : ', rec_CNN_inc_aug_n)

### CONFUSION MATRIX
cm_CNN_aug_c = confusion_matrix(np.argmax(y_te, axis=-1), np.argmax(CNN_aug.classifier.predict(x_te), axis=-1), labels=[i for i in range(12)], sample_weight=None)
cm_CNN_aug_n = confusion_matrix(np.argmax(y_te_n, axis=-1), np.argmax(CNN_aug.classifier.predict(x_te_n), axis=-1), labels=[i for i in range(12)], sample_weight=None)
cm_CNN_inc_aug_c = confusion_matrix(np.argmax(y_te, axis=-1), np.argmax(CNN_inc_aug.classifier.predict(x_te), axis=-1), labels=[i for i in range(12)], sample_weight=None)
cm_CNN_inc_aug_n = confusion_matrix(np.argmax(y_te_n, axis=-1), np.argmax(CNN_inc_aug.classifier.predict(x_te_n), axis=-1), labels=[i for i in range(12)], sample_weight=None)

