# Import Packages
from keras.models import Model, load_model, model_from_json
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Reshape, UpSampling2D, concatenate
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
import os

class MyModel(object):
	def __init__(self, x_tr, y_tr, x_te, y_te, kind, augmentation=False):
		self.kind = kind
		self.augmentation = augmentation
		self.n_classes = y_tr.shape[1]
		self.classifier = None
		self.encoder = None
		self.autoencoder = None
		self.x_tr, self.y_tr, self.x_te, self.y_te = x_tr, y_tr, x_te, y_te
		self.feat_tr, self.feat_te = None, None


	def init(self, lr=0.001, feat_dim=256):

		# CNN Model
		if self.kind == 'CNN':
			self.classifier = self.get_CNN()

		# CNN AE + DFNN Model
		if self.kind == 'CNN_AE':

			# CNN AE
			self.encoder, self.autoencoder = self.get_CNN_AE(feat_dim=feat_dim)

			# DFNN
			self.classifier = self.get_DFNN(feat_dim=feat_dim)

		# FNN AE + DFNN
		if self.kind == 'FNN_AE':

			# FNN AE
			self.encoder, self.autoencoder = self.get_FNN_AE(feat_dim=feat_dim)

			# DFNN
			self.classifier = self.get_DFNN(feat_dim=feat_dim)
            
		# CNN inception
		if self.kind == 'CNN_inc':
			self.classifier = self.get_CNN_inc()

		return

	def train_classifier(self, epochs=50, batch_size=128):

		if self.kind == 'CNN':
			kind = self.kind
			if self.augmentation:
				kind = kind+'_aug'

			tensorboard = TensorBoard(log_dir='tensorboard/'+kind, histogram_freq=0, write_graph=True, write_images=True)
			if not os.path.exists('models_backup'):
				os.makedirs('models_backup')
			checkpoint = ModelCheckpoint('models_backup/'+kind+'.hdf5',
										monitor='val_acc',
										verbose=1,
										save_best_only=True,
										mode='max')

			self.classifier.fit(self.x_tr, self.y_tr,
					epochs=epochs,
					batch_size=batch_size,
					shuffle=True,
					validation_data=(self.x_te, self.y_te),
					callbacks=[tensorboard, checkpoint])

			self.load_best_model()

		if self.kind == 'CNN_AE':

			# Get the Features
			self.feat_tr = self.encoder.predict(self.x_tr).reshape(self.x_tr.shape[0], -1)
			self.feat_te = self.encoder.predict(self.x_te).reshape(self.x_te.shape[0], -1)

			kind = self.kind
			if self.augmentation:
				kind = kind+'_aug'

			tensorboard = TensorBoard(log_dir='tensorboard/DFNN_'+kind, histogram_freq=0, write_graph=True, write_images=True)
			if not os.path.exists('models_backup'):
				os.makedirs('models_backup')
			checkpoint = ModelCheckpoint('models_backup/DFNN_'+kind+'.hdf5',
										monitor='val_acc',
										verbose=1,
										save_best_only=True,
										mode='max')

			self.classifier.fit(self.feat_tr, self.y_tr,
					epochs=epochs,
					batch_size=batch_size,
					shuffle=True,
					validation_data=(self.feat_te, self.y_te),
					callbacks=[tensorboard, checkpoint])

			self.load_best_model(kind_partial='DFNN')
            
		if self.kind == 'FNN_AE':

			# Get the Features
			self.feat_tr = self.encoder.predict(self.x_tr.reshape(-1, 99*40)).reshape(self.x_tr.shape[0], -1)
			self.feat_te = self.encoder.predict(self.x_te.reshape(-1, 99*40)).reshape(self.x_te.shape[0], -1)

			kind = self.kind
			if self.augmentation:
				kind = kind+'_aug'

			tensorboard = TensorBoard(log_dir='tensorboard/DFNN_'+kind, histogram_freq=0, write_graph=True, write_images=True)
			if not os.path.exists('models_backup'):
				os.makedirs('models_backup')
			checkpoint = ModelCheckpoint('models_backup/DFNN_'+kind+'.hdf5',
										monitor='val_acc',
										verbose=1,
										save_best_only=True,
										mode='max')

			self.classifier.fit(self.feat_tr, self.y_tr,
					epochs=epochs,
					batch_size=batch_size,
					shuffle=True,
					validation_data=(self.feat_te, self.y_te),
					callbacks=[tensorboard, checkpoint])

			self.load_best_model(kind_partial='DFNN')
        
		if self.kind == 'CNN_inc':
			kind = self.kind
			if self.augmentation:
				kind = kind+'_aug'

			tensorboard = TensorBoard(log_dir='tensorboard/'+kind, histogram_freq=0, write_graph=True, write_images=True)
			if not os.path.exists('models_backup'):
				os.makedirs('models_backup')
			checkpoint = ModelCheckpoint('models_backup/'+kind+'.hdf5',
										monitor='val_acc',
										verbose=1,
										save_best_only=True,
										mode='max')

			self.classifier.fit(self.x_tr, self.y_tr,
					epochs=epochs,
					batch_size=batch_size,
					shuffle=True,
					validation_data=(self.x_te, self.y_te),
					callbacks=[tensorboard, checkpoint])

			self.load_best_model()
            
		return

	def train_autoencoder(self, epochs=50, batch_size=128):
		if self.kind == 'FNN_AE':
			self.x_tr = self.x_tr.reshape(-1, 99*40)
			self.x_te = self.x_te.reshape(-1, 99*40)
        
		kind = self.kind
		if self.augmentation:
			kind = kind+'_aug'

		tensorboard = TensorBoard(log_dir='tensorboard/'+kind, histogram_freq=0, write_graph=True, write_images=True)
		if not os.path.exists('models_backup'):
			os.makedirs('models_backup')
		checkpoint = ModelCheckpoint('models_backup/'+kind+'.hdf5',
									monitor='val_loss',
									verbose=1,
									save_best_only=True,
									mode='min')

		self.autoencoder.fit(self.x_tr, self.x_tr,
				epochs=epochs,
				batch_size=batch_size,
				shuffle=True,
				validation_data=(self.x_te, self.x_te),
				callbacks=[tensorboard, checkpoint])

		self.load_best_model(kind_partial='AE')

		return


	def load_best_model(self, lr=0.001, feat_dim=256, kind_partial=None):

		# CNN
		if self.kind == 'CNN':
			CNN = self.get_CNN()
			kind = self.kind
			if self.augmentation:
				kind = kind+'_aug'
			CNN.load_weights('models_backup/'+kind+'.hdf5')
			self.classifier = CNN

		# CNN AE + DFNN
		if self.kind == 'CNN_AE':
			if kind_partial == 'AE':
				# CNN AE
				encoder, autoencoder = self.get_CNN_AE(feat_dim=feat_dim)

				kind = self.kind
				if self.augmentation:
					kind = kind+'_aug'
				autoencoder.load_weights('models_backup/'+kind+'.hdf5')
				encoder.set_weights(autoencoder.get_weights()[:9])
				self.autoencoder = autoencoder
				self.encoder = encoder

			if kind_partial == 'DFNN':
				# DFNN
				DFNN = self.get_DFNN(feat_dim=feat_dim)
				kind = self.kind
				if self.augmentation:
					kind = kind+'_aug'
				DFNN.load_weights('models_backup/DFNN_'+kind+'.hdf5')
				self.classifier = DFNN

		# FNN AE + DFNN
		if self.kind == 'FNN_AE':
			if kind_partial == 'AE':
				# FNN AE
				encoder, autoencoder = self.get_FNN_AE(feat_dim=feat_dim)

				kind = self.kind
				if self.augmentation:
					kind = kind+'_aug'
				autoencoder.load_weights('models_backup/'+kind+'.hdf5')
				encoder.set_weights(autoencoder.get_weights()[:2])
				self.autoencoder = autoencoder
				self.encoder = encoder

			if kind_partial == 'DFNN':
				# DFNN
				DFNN = self.get_DFNN(feat_dim=feat_dim)
				kind = self.kind
				if self.augmentation:
					kind = kind+'_aug'
				DFNN.load_weights('models_backup/DFNN_'+kind+'.hdf5')
				self.classifier = DFNN
                
		# CNN
		if self.kind == 'CNN_inc':
			CNN_inc = self.get_CNN_inc()
			kind = self.kind
			if self.augmentation:
				kind = kind+'_aug'
			CNN_inc.load_weights('models_backup/'+kind+'.hdf5')
			self.classifier = CNN_inc

		return


	def get_CNN(self):
	
		input_shape = [99,40,1]
		input_img = Input(shape=input_shape)

		depth = 32
		x = Conv2D(1*depth, (5, 5), padding='same', activation='relu', input_shape=input_shape)(input_img)
		x = MaxPooling2D(pool_size=(2, 1))(x)
		x = Dropout(0.25)(x)
		 
		x = Conv2D(2*depth, (7, 5), activation='relu')(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)
		x = Dropout(0.25)(x)

		x = Conv2D(4*depth, (7, 5), activation='relu')(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)
		x = Dropout(0.25)(x)
		 
		x = Flatten()(x)
		x = Dense(128, activation='relu')(x)
		x = Dropout(0.5)(x)
		x = Dense(self.n_classes, activation='softmax')(x)

		CNN = Model(input_img, x)
		opt = Adam(lr=0.001)
		CNN.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

		return CNN

	def get_CNN_AE(self, feat_dim=256):
		input_img = Input(shape=(99, 40, 1))  # adapt this if using `channels_first` image data format
		depth = 32
		x = Conv2D(1*depth, (3, 3), activation='relu', padding='same')(input_img)
		x = MaxPooling2D((2, 2), padding='same')(x)
		x = Conv2D(2*depth, (3, 3), activation='relu', padding='same')(x)
		x = MaxPooling2D((2, 1), padding='same')(x)
		x = Conv2D(4*depth, (3, 3), activation='relu', padding='same')(x)
		x = MaxPooling2D((3, 2), padding='same')(x)
		x = Flatten()(x)
		encoded = Dense(feat_dim, activation='relu')(x)

		x = Dense(9*7*4*depth, activation='relu')(encoded)
		x = Reshape((9,7,4*depth))(x)
		x = Conv2D(4*depth, (3, 3), activation='relu', padding='same')(x)
		x = UpSampling2D((3, 2))(x)
		x = Conv2D(2*depth, (5, 3), activation='relu', padding='same')(x)
		x = UpSampling2D((2, 2))(x)
		x = Conv2D(1*depth, (3, 6), activation='relu')(x)
		x = UpSampling2D((2, 2))(x)
		decoded = Conv2D(1, (6, 7), activation='sigmoid', padding='valid')(x)

		encoder = Model(input_img, encoded)
		autoencoder = Model(input_img, decoded)
		opt = Adam(lr=0.001)
		autoencoder.compile(optimizer=opt, loss='mse')

		return encoder, autoencoder

	def get_DFNN(self, feat_dim=256):
		input_feature = Input(shape=(feat_dim,))
		x = Dense(512, activation='relu')(input_feature)
		x = Dropout(0.3)(x)
		x = Dense(64, activation='relu')(x)
		x = Dropout(0.3)(x)
		y_ = Dense(self.n_classes, activation='softmax')(x)

		DFNN = Model(input_feature, y_)
		opt = Adam(lr=0.001)
		DFNN.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
		return DFNN

	def get_FNN_AE(self, feat_dim=256):
		x_in = Input(shape=(99*40,))
		encoded = Dense(feat_dim, activation='relu')(x_in)

		decoded = Dense(99*40, activation='sigmoid')(encoded)

		encoder = Model(x_in, encoded)
		autoencoder = Model(x_in, decoded)
		opt_AE = Adam(lr=0.001)
		autoencoder.compile(optimizer=opt_AE, loss='mse')
		return encoder, autoencoder

	def get_CNN_inc(self):
		input_shape = [99,40,1]
		input_img = Input(shape=input_shape)

		n_ch = 64
		x = Conv2D(n_ch, (5, 5), strides=[1,1], activation='relu', input_shape=input_shape)(input_img)
		x = MaxPooling2D(pool_size=(3, 2))(x)

		tower_1 = Conv2D(n_ch, (1,1), padding='same', activation='relu')(x)
		tower_1 = Conv2D(n_ch, (5,5), padding='same', activation='relu')(tower_1)
		tower_2 = Conv2D(n_ch, (1,1), padding='same', activation='relu')(x)
		tower_2 = Conv2D(n_ch, (7,7), padding='same', activation='relu')(tower_2)
		tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
		tower_3 = Conv2D(n_ch, (1,1), padding='same', activation='relu')(tower_3)
		output = concatenate([tower_1, tower_2, tower_3], axis = 3)

		x = Conv2D(n_ch, (15, 2), strides=[1,1], activation='relu')(output)
		x = MaxPooling2D(pool_size=(2, 2))(x)

		output = Flatten()(x)
		out = Dense(512, activation='relu')(output)
		out = Dropout(0.3)(out)
		out = Dense(self.n_classes, activation='softmax')(out)

		CNN_inc = Model(input_img, out)
		opt_inc = Adam(lr=0.001)
		CNN_inc.compile(loss='binary_crossentropy', optimizer=opt_inc, metrics=['accuracy'])
		return CNN_inc