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
import tensorboardX


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epoch_num = 200
lr = 0.01
#net = VGGNet().to(device)
net = InceptionNetSmall().to(device)
#loss
loss_fun = nn.CrossEntropyLoss()
#optim
optimizer = torch.optim.Adam(net.parameters(),lr = lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = 5,gamma = 0.9)

if not os.path.exists('F:/DownloadFile/cifar-10-batches-py/log3'):
		os.mkdir('F:/DownloadFile/cifar-10-batches-py/log3')

writer = tensorboardX.SummaryWriter('F:/DownloadFile/cifar-10-batches-py/log3')

#train
step_n = 0
for epoch in range(epoch_num):
	print("epoch is ", epoch)
	
	for idx,data in enumerate(train_loader):
		net.train()
		inputs,labels = data
		inputs,labels = inputs.to(device),labels.to(device)
		outputs = net(inputs)
		batch_size = inputs.size(0)
		loss = loss_fun(outputs,labels)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		_,pred = torch.max(outputs,dim = 1)
		correct = pred.eq(labels.data).cpu().sum()
		print("train step",idx,"loss is ",loss.item(),"mini-batch correct is ",100.0 * correct / 200)
		writer.add_scalar('train loss',loss.item(),global_step = step_n)
		writer.add_scalar('train correct',100.0 * correct.item() / 200,global_step = step_n)
		step_n += 1
	if not os.path.exists('F:/DownloadFile/cifar-10-batches-py/models'):
		os.mkdir('F:/DownloadFile/cifar-10-batches-py/models')
	torch.save(net.state_dict(),'F:/DownloadFile/cifar-10-batches-py/models/{}.pth'.format(epoch + 1))
	scheduler.step()
	print("train lr is ",optimizer.state_dict()['param_groups'][0]['lr'])


	sum_loss = 0
	sum_correct = 0
	for idx,data in enumerate(test_loader):
		net.eval()
		inputs,labels = data
		inputs,labels = inputs.to(device),labels.to(device)
		outputs = net(inputs)
		batch_size = inputs.size(0)
		loss = loss_fun(outputs,labels)
		_,pred = torch.max(outputs,dim = 1)
		correct = pred.eq(labels.data).cpu().sum()
		sum_loss += loss.item()
		sum_correct += correct.item()
		
		#print("test step",idx,"loss is ",loss.item(),"mini-batch correct is ",100.0 * correct / 168)
	test_loss = sum_loss * 1.0 / len(test_loader)
	test_correct = sum_correct * 100.0 / len(test_loader) / 200
	writer.add_scalar('test loss',test_loss,global_step = epoch + 1)
	writer.add_scalar('test correct',test_correct,global_step = epoch + 1)
	print("epoch step",epoch + 1,"test loss is ",test_loss,"mini-batch correct is ",test_correct)
writer.close()