import numpy as np
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


label_name = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
label_dict = {}
for idx,name in enumerate(label_name):
	label_dict[name] = idx

def default_loader(path):
	return Image.open(path).convert('RGB')

train_transform = transforms.Compose([
	transforms.RandomResizedCrop((28,28)),
	transforms.RandomHorizontalFlip(),
	transforms.RandomVerticalFlip(),
	transforms.RandomRotation(90),
	transforms.RandomGrayscale(0.1),
	transforms.ColorJitter(0.3,0.3,0.3,0.3),
	transforms.ToTensor()
	])
test_transform = transforms.Compose([
	transforms.Resize((28,28)),
	transforms.ToTensor()
	])

class MyDataSet(Dataset):
	def __init__(self,im_list,transform = None,loader = default_loader):
		super(MyDataSet,self).__init__()
		img = []
		for i in im_list:
			im_label_name = i.split("\\")[-2]
			img.append([i,label_dict[im_label_name]])
		self.imgs = img
		self.transform = transform
		self.loader = loader
	def __getitem__(self,index):
		im_path,im_label = self.imgs[index]
		im_data = self.loader(im_path)
		if self.transform is not None:
			im_data = self.transform(im_data)
		return im_data,im_label
	def __len__(self):
		return len(self.imgs)

im_train_list = glob.glob(r'F:\DownloadFile\cifar-10-batches-py\train\*\*.png')
im_test_list = glob.glob(r'F:\DownloadFile\cifar-10-batches-py\test\*\*.png')

train_dataset = MyDataSet(im_train_list,transform = train_transform)
test_dataset = MyDataSet(im_test_list,transform = test_transform)

train_loader = DataLoader(dataset = train_dataset,
							batch_size= 200,
							shuffle = True
							)
test_loader = DataLoader(dataset = test_dataset,
							batch_size= 200,
							shuffle = False
							)

class VGGbase(nn.Module):
	def __init__(self):
		super(VGGbase,self).__init__()
		#28*28*3
		self.conv1 = nn.Sequential(
			nn.Conv2d(3,64,kernel_size = 3,stride = 1,padding = 1),
			nn.BatchNorm2d(64),
			nn.ReLU()
			)
		self.max_pooling1 = nn.MaxPool2d(kernel_size = 2,stride = 2)
		#14*14*64
		self.conv2_1 = nn.Sequential(
			nn.Conv2d(64,128,kernel_size = 3,stride = 1,padding = 1),
			nn.BatchNorm2d(128),
			nn.ReLU()
			)
		self.conv2_2 = nn.Sequential(
			nn.Conv2d(128,128,kernel_size = 3,stride = 1,padding = 1),
			nn.BatchNorm2d(128),
			nn.ReLU()
			)
		self.max_pooling2 = nn.MaxPool2d(kernel_size = 2,stride = 2)
		#7*7*128
		self.conv3_1 = nn.Sequential(
			nn.Conv2d(128,256,kernel_size = 3,stride = 1,padding = 1),
			nn.BatchNorm2d(256),
			nn.ReLU()
			)
		self.conv3_2 = nn.Sequential(
			nn.Conv2d(256,256,kernel_size = 3,stride = 1,padding = 1),
			nn.BatchNorm2d(256),
			nn.ReLU()
			)
		self.max_pooling3 = nn.MaxPool2d(kernel_size = 2,stride = 2,padding = 1)
		#4*4*256
		self.conv4_1 = nn.Sequential(
			nn.Conv2d(256,512,kernel_size = 3,stride = 1,padding = 1),
			nn.BatchNorm2d(512),
			nn.ReLU()
			)
		self.conv4_2 = nn.Sequential(
			nn.Conv2d(512,512,kernel_size = 3,stride = 1,padding = 1),
			nn.BatchNorm2d(512),
			nn.ReLU()
			)
		self.max_pooling4 = nn.MaxPool2d(kernel_size = 2,stride = 2)
		#2*2*512
		self.fc = nn.Linear(512 * 2 * 2,10)

	def forward(self,x):
		batch_size = x.size(0)
		out = self.conv1(x)
		out = self.max_pooling1(out)
		out = self.conv2_1(out)
		out = self.conv2_2(out)
		out = self.max_pooling2(out)
		out = self.conv3_1(out)
		out = self.conv3_2(out)
		out = self.max_pooling3(out)
		out = self.conv4_1(out)
		out = self.conv4_2(out)
		out = self.max_pooling4(out)
		out = out.view(batch_size,-1)
		out = self.fc(out)
		out = F.log_softmax(out,dim = 1)
		return out

def VGGNet():
	return VGGbase()

class ResBlock(nn.Module):
	def __init__(self,in_channel,out_channel,stride = 1):
		super(ResBlock,self).__init__()
		self.layer = nn.Sequential(
			nn.Conv2d(in_channel,out_channel,
				kernel_size = 3,stride = stride,padding = 1),
			nn.BatchNorm2d(out_channel),
			nn.ReLU(),
			nn.Conv2d(out_channel,out_channel,
				kernel_size = 3,stride = 1,padding = 1),
			nn.BatchNorm2d(out_channel)
			)
		self.shortcut = nn.Sequential()
		if in_channel != out_channel or stride > 1:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_channel,out_channel,
					kernel_size = 3,stride = stride,padding = 1),
				nn.BatchNorm2d(out_channel)
				)
	def forward(self,x):
		out1 = self.layer(x)
		out2 = self.shortcut(x)
		out = out1 + out2
		out = F.relu(out)
		return out

class ResNet(nn.Module):
	def make_layer(self,block,out_channel,stride,num_block):
		layer_list = []
		for i in range(num_block):
			if i == 0:
				in_stride = stride
			else:
				in_stride = 1
			layer_list.append(block(self.in_channel,out_channel,in_stride))
			self.in_channel = out_channel
		return nn.Sequential(*layer_list)

	def __init__(self):
		super(ResNet,self).__init__()
		self.in_channel = 32
		self.conv1 = nn.Sequential(
			nn.Conv2d(3,32,kernel_size = 3,stride = 1,padding = 1),
			nn.BatchNorm2d(32),
			nn.ReLU()
			)
		self.layer1 = self.make_layer(ResBlock,64,2,2)
		self.layer2 = self.make_layer(ResBlock,128,2,2)
		self.layer3 = self.make_layer(ResBlock,256,2,2)
		self.layer4 = self.make_layer(ResBlock,512,2,2)
		self.fc = nn.Linear(512,10)
	def forward(self,x):
		out = self.conv1(x)
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = F.avg_pool2d(out,2)
		out = out.view(out.size(0),-1)
		out = self.fc(out)
		return out

def resnet():
	return ResNet()


class mobileNet(nn.Module):
	def conv_dw_pw(self,in_channel,out_channel,stride):
		return nn.Sequential(
			nn.Conv2d(in_channel,in_channel,kernel_size = 3,stride = stride,padding = 1,groups = in_channel,bias = False),
			nn.BatchNorm2d(in_channel),
			nn.ReLU(),
			nn.Conv2d(in_channel,out_channel,kernel_size = 1,stride = 1,padding = 0,bias = False),
			nn.BatchNorm2d(out_channel),
			nn.ReLU()
			)
	def __init__(self):
		super(mobileNet,self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(3,32,kernel_size = 3,stride = 1,padding = 1),
			nn.BatchNorm2d(32),
			nn.ReLU()
			)
		self.conv_dw_pw1 = self.conv_dw_pw(32,32,1)
		self.conv_dw_pw2 = self.conv_dw_pw(32,64,2)
		self.conv_dw_pw3 = self.conv_dw_pw(64,64,1)
		self.conv_dw_pw4 = self.conv_dw_pw(64,128,2)
		self.conv_dw_pw5 = self.conv_dw_pw(128,128,1)
		self.conv_dw_pw6 = self.conv_dw_pw(128,256,2)
		self.conv_dw_pw7 = self.conv_dw_pw(256,256,1)
		self.conv_dw_pw8 = self.conv_dw_pw(256,512,2)
		self.fc = nn.Linear(512,10)
	def forward(self,x):
		out = self.conv1(x)
		out = self.conv_dw_pw1(out)
		out = self.conv_dw_pw2(out)
		out = self.conv_dw_pw3(out)
		out = self.conv_dw_pw4(out)
		out = self.conv_dw_pw5(out)
		out = self.conv_dw_pw6(out)
		out = self.conv_dw_pw7(out)
		out = self.conv_dw_pw8(out)
		out = F.avg_pool2d(out,2)
		out = out.view(-1,512)
		out = self.fc(out)
		return out

def mobilenet():
	return mobileNet()

def convBNReLU(in_channel,out_channel,kernel_size):
	return nn.Sequential(
		nn.Conv2d(in_channel,out_channel,
			kernel_size = kernel_size,
			stride = 1,
			padding = kernel_size // 2),
		nn.BatchNorm2d(out_channel),
		nn.ReLU()
		)

class InceptionNetBase(nn.Module):
	def __init__(self,in_channel,out_channel_list,reduce_channel_list):
		super(InceptionNetBase,self).__init__()
		self.branch1_conv = convBNReLU(in_channel,out_channel_list[0],1)
		self.branch2_conv1 = convBNReLU(in_channel,reduce_channel_list[0],1)
		self.branch2_conv2 = convBNReLU(reduce_channel_list[0],out_channel_list[1],3)
		self.branch3_conv1 = convBNReLU(in_channel,reduce_channel_list[1],1)
		self.branch3_conv2 = convBNReLU(reduce_channel_list[1],out_channel_list[2],5)
		self.branch4_pool = nn.MaxPool2d(kernel_size = 3,stride = 1,padding = 1)
		self.branch4_conv = convBNReLU(in_channel,out_channel_list[3],3)
	def forward(self,x):
		out1 = self.branch1_conv(x)
		out2 = self.branch2_conv1(x)
		out2 = self.branch2_conv2(out2)
		out3 = self.branch3_conv1(x)
		out3 = self.branch3_conv2(out3)
		out4 = self.branch4_pool(x)
		out4 = self.branch4_conv(out4)
		out = torch.cat([out1,out2,out3,out4],dim = 1)
		return out

class InceptionNet(nn.Module):
	def __init__(self):
		super(InceptionNet,self).__init__()
		self.block1 = nn.Sequential(
			nn.Conv2d(3,64,
				kernel_size = 7,
				stride = 2,
				padding = 1),
			nn.BatchNorm2d(64),
			nn.ReLU()
			)
		self.block2 = nn.Sequential(
			nn.Conv2d(64,128,
				kernel_size = 3,
				stride = 2,
				padding = 1),
			nn.BatchNorm2d(128),
			nn.ReLU()
			)
		self.block3 = nn.Sequential(
			InceptionNetBase(in_channel = 128,
				out_channel_list = [64,64,64,64],
				reduce_channel_list = [16,16]),
			nn.MaxPool2d(kernel_size = 3,stride = 2,padding = 1)
			)
		self.block4 = nn.Sequential(
			InceptionNetBase(in_channel = 256,
				out_channel_list = [96,96,96,96],
				reduce_channel_list = [32,32]),
			nn.MaxPool2d(kernel_size = 3,stride = 2,padding = 1)
			)
		self.fc = nn.Linear(384,10)

	def forward(self,x):
		out = self.block1(x)
		out = self.block2(out)
		out = self.block3(out)
		out = self.block4(out)
		out = F.avg_pool2d(out,2)
		out = out.view(out.size(0),-1)
		out = self.fc(out)
		return out

def InceptionNetSmall():
	return InceptionNet()