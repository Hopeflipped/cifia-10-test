import pickle
import glob
import numpy as np
import cv2
import os

def unpickle(file):
	with open(file,'rb') as f:
		dict1 = pickle.load(f,encoding = 'bytes')
	return dict1

label_name = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
save_path = 'F:/DownloadFile/cifar-10-batches-py/test'

train_list = glob.glob('F:/DownloadFile/cifar-10-batches-py/test_batch')
print(train_list)
for i in train_list:
	l_list = unpickle(i)
	#print(l_list.keys())
	for idx,idata in enumerate(l_list[b'data']):
		im_label = l_list[b'labels'][idx]
		im_name = l_list[b'filenames'][idx]
		#print(idx,im_label,im_name,idata)
		im_label_name = label_name[im_label]
		im_data = np.reshape(idata,(3,32,32))
		im_data = np.transpose(im_data,(1,2,0))
		if not os.path.exists('{}/{}'.format(save_path,im_label_name)):
			os.mkdir('{}/{}'.format(save_path,im_label_name))
		cv2.imwrite('{}/{}/{}'.format(save_path,im_label_name,im_name.decode('utf-8')),im_data)