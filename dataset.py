# Import Packages
import os
import numpy as np
from IPython.display import clear_output

class MyDataset(object):
    def __init__(self, data_dir, augmented=False):
        self.augmented = augmented
        if augmented:
            self.data_dir = data_dir+'_aug'
        else:
            self.data_dir = data_dir
        
        self.classes_names, self.y_file_names, self.y_class_names, self.n_classes, self.n_samples, self.n_samples_per_class = self.get_dataset_info()
        self.data = self.get_data()
        self.N_tr, self.N_te = self.data[0].shape[0], self.data[2].shape[0]


    def get_dataset_info(self, print_info=True):

        # Directory of the Dataset
        data_dir = self.data_dir
        classes_names = [cl for cl in os.listdir(data_dir) if os.path.isdir(data_dir+'/'+cl)]
        classes_names.sort()
        y_file_names = []
        y_class_names = []
        n_classes = len(classes_names)

        # Compute the Number of Samples in the Dataset
        n_samples = 0
        n_samples_per_class = {}
        for i,cl in enumerate(classes_names):
            n_samples_per_class_temp = 0
            file_names = os.listdir(data_dir+'/'+cl)
            file_names.sort()
            for a in file_names:
                if a[-4:] == '.npy':
                    n_samples += 1
                    n_samples_per_class_temp += 1
                    y_file_names.append(cl+'/'+a)
                    y_class_names.append(cl)
            n_samples_per_class[cl] = n_samples_per_class_temp

        return classes_names, y_file_names, y_class_names, n_classes, n_samples, n_samples_per_class
    
    def info(self):
        
        # Print Info of the Dataset
        print('N Classes :', self.n_classes)
        print('Class / Samples:')
        for key in self.n_samples_per_class.keys():
            print('-', key, ':', self.n_samples_per_class[key])
        print('N Sampeles Tot :', self.n_samples)
        return

    def get_data(self):
        y = np.zeros((self.n_samples, self.n_classes))
        MFSC = np.zeros((self.n_samples,99,40,1))

        # Load the Dataset
        count = 0
        for i,cl in enumerate(self.classes_names):
            clear_output(wait=True)
            print('Load Dataset :', 100.*(i+1)/self.n_classes,'%')
            file_names = os.listdir(self.data_dir+'/'+cl)
            file_names.sort()
            for a in file_names:
                if a[-4:] == '.npy':
                    MFSC[count,:,:,:] = np.load(self.data_dir+'/'+cl+'/'+a)
                    y[count, i] = 1
                    count += 1
        clear_output(wait=True)
        print('Dataset Loaded.')

        # Train - Test Split
        N_tr = int(0.7 * self.n_samples)
        N_te = self.n_samples - N_tr
        idx_tr = np.arange(self.n_samples)
        np.random.seed(seed=0)
        np.random.shuffle(idx_tr)
        idx_tr = idx_tr[:N_tr]
        idx_te = list(set([i for i in range(self.n_samples)]) - set(idx_tr))

        x_tr = MFSC[idx_tr,:,:,:]
        y_tr = y[idx_tr,:]
        x_te = MFSC[idx_te,:,:,:]
        y_te = y[idx_te,:]

        print('Train Size :', N_tr)
        print('Test Size :', N_te)

        return x_tr, y_tr, x_te, y_te

