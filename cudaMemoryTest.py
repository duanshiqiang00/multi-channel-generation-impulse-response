#
# '''
# 来源于
# TimeAutoML: Autonomous Representation Learning for Multivariate Irregularly Sampled Time Series
# 通过阻尼和频率参数表征计算阻尼和频率参数
# '''
# import os
# import numpy
# import torch
# from sklearn.model_selection import train_test_split
# import torch.utils.data as Data
# import matplotlib.pyplot as plt
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# import random
# from scipy.fftpack import fft
# import math
#
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# class dataAbout():
#     def load_data(dataPath, timeSignalLength):
#         dataFileNameLists = os.listdir(dataPath);
#         originalSignaData = numpy.empty((len(dataFileNameLists), 1, timeSignalLength), dtype='float64');
#         originalSignalFreq=numpy.empty((len(dataFileNameLists),1),dtype='float64')
#         # addNoiseSignalData = numpy.empty((len(dataFileNameLists), 1, timeSignalLength), dtype='float64');
#
#         for dataFileName in dataFileNameLists:
#             dataFilePathName = dataPath + "/" + dataFileName
#
#             allDataFile = open(dataFilePathName)
#             allData = numpy.loadtxt(allDataFile)
#
#             allDataMean = numpy.mean(allData, axis=0)
#             allDatastd = numpy.std(allData, axis=0)
#             allData = (allData - allDataMean) / allDatastd
#
#             originalSignalFreq[dataFileNameLists.index(dataFileName),:]= float(dataFileName.split('_')[0].split('-')[1])
#             originalSignaData[dataFileNameLists.index(dataFileName), :, :] = numpy.reshape(allData.T,
#                                                                                            (1, timeSignalLength));
#             # addNoiseSignalData[dataFileNameLists.index(dataFileName), :, :] = numpy.reshape(allData[:, 1].T,
#             #                                                                                 (1, timeSignalLength));
#             #
#             # print("dataFileName.split('_')[0]:",dataFileName.split('_')[0])
#             # print("dataFileName.split('_')[0].split('-')[1]:",dataFileName.split('_')[0].split('-')[1])
#             # print(originalSignalFreq)
#         return originalSignaData,originalSignalFreq
#
#     def load_real_data(realDataPath):
#         realDataFileNameLists = os.listdir(realDataPath);
#         realDataTimeSeriesList = []
#
#         for realDataFileName in realDataFileNameLists:
#             realDataFilePathName = realDataPath + "/" + realDataFileName
#             realDataFile = open(realDataFilePathName)
#             realDataTimeSeries = numpy.loadtxt(realDataFile)
#             realDataTimeSeriesMean = numpy.mean(realDataTimeSeries, axis=0)
#             realDataTimeSeriesStd = numpy.std(realDataTimeSeries, axis=0)
#             realDataTimeSeries = (realDataTimeSeries - realDataTimeSeriesMean) / realDataTimeSeriesStd
#             realDataTimeSeriesList.append(realDataTimeSeries)
#         return realDataFileNameLists, realDataTimeSeriesList
#
#     def self_train_test_split(ALlData, ALlLabel, TRAIN_TEST_RATE):
#         TrainData, TestData, TrainLabel, TestLabel \
#             = train_test_split(ALlData[:MAXDATASIZE, :, :], ALlLabel[:MAXDATASIZE, :], test_size=TRAIN_TEST_RATE,
#                                shuffle=True)
#         ## 此处MAXDATASIZE 表示读入 的最大数据量
#         # = train_test_split(ALlData[:MAXDATASIZE, :, :, :], ALlLabel[:MAXDATASIZE], test_size=TRAIN_TEST_RATE)
#
#         return TrainData, TestData, TrainLabel, TestLabel
#
#     def numpyTOFloatTensor(data):
#         data = torch.from_numpy(data)
#         tensorData = torch.FloatTensor.float(data)
#         return tensorData
#
#     def numpyTOLongTensor(data):
#         data = torch.from_numpy(data)
#         tensorData = torch.LongTensor.long(data)
#         return tensorData
#
#     def modelChoice(model, data1, data2, label):
#
#         if model == 'STFT':
#             return data1, label
#         if model == 'EMD':
#             return data2, label
#         if model == 'STFT+EMD':
#             data = numpy.concatenate((data1, data2), axis=1)
#             return data, label
#         else:
#             print("(口..口) 没有输入合适的CNN输入模式 (口..口)")
#             exit()
#
#     def NetInputLayerNum(model):
#         if model == 'STFT':
#             return 3
#         if model == 'EMD':
#             return 1
#         if model == 'STFT+EMD':
#             return 4
#         else:
#             print("(口..口) 没有输入合适的CNN输入层的数量 (口..口)")
#             exit()
#
#     # 数据封装函数
#     def data_loader(data_x, data_y):
#
#         train_data = Data.TensorDataset(data_x, data_y)
#         train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)
#         return train_loader
#
#     def mergeThreeList(List1, List2, List3):
#
#         List1Size = len(List1)
#         List2Size = len(List2)
#         List3Size = len(List3)
#
#         arrayList1 = numpy.array(List1)
#         arrayList1 = arrayList1.reshape(List1Size, 1)
#         arrayList2 = numpy.array(List2)
#         arrayList2 = arrayList2.reshape(List2Size, 1)
#         arrayList3 = numpy.array(List3)
#         arrayList3 = arrayList3.reshape(List3Size, 1)
#         mergedArrayList = numpy.concatenate((arrayList1, arrayList2, arrayList3), axis=1)
#         return mergedArrayList
#
#     def mergeTwoList(List1, List2):
#
#         List1Size = len(List1)
#         List2Size = len(List2)
#
#         arrayList1 = numpy.array(List1)
#         arrayList1 = arrayList1.reshape(List1Size, 1)
#         arrayList2 = numpy.array(List2)
#         arrayList2 = arrayList2.reshape(List2Size, 1)
#         mergedArrayList = numpy.concatenate((arrayList1, arrayList2), axis=1)
#         return mergedArrayList
#
#
# def mkSaveModelResultdir(path):
#     folder = os.path.exists(path)
#     if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
#         os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
#         return path
#     else:
#         return path
#
#
# def calculateSpectrumLoss(dataReal1, dataReal2):
#     dataImage1 = torch.zeros([dataReal1.shape[0], dataReal1.shape[1], dataReal1.shape[2]])
#     dataImage1 = dataImage1.cuda().float()
#     data1 = torch.cat((dataReal1, dataImage1), dim=1)
#     data1 = data1.view(data1.shape[0], data1.shape[2], data1.shape[1])
#     data1fft = torch.fft(data1, signal_ndim=1)
#     data1fftabs = torch.abs(data1fft)
#     data1log = torch.log10(torch.add(data1fftabs, 0.1))
#
#     dataImage2 = torch.zeros([dataReal2.shape[0], dataReal2.shape[1], dataReal2.shape[2]])
#     dataImage2 = dataImage2.cuda().float()
#     data2 = torch.cat((dataReal2, dataImage2), dim=1)
#     data2 = data2.view(data2.shape[0], data2.shape[2], data2.shape[1])
#     data2fft = torch.fft(data2, signal_ndim=1)
#     data2fftabs = torch.abs(data2fft)
#     data2log = torch.log10(torch.add(data2fftabs, 0.1))
#
#     return data1log, data2log
#
#
# ########################################################################################################################
# class flutterFilterNet(nn.Module):
#     def __init__(self):
#         super(flutterFilterNet, self).__init__()
#
#         self.encoder = nn.Sequential(
#             nn.Conv1d(in_channels=1, out_channels=2, kernel_size=[3], stride=1, padding=1),
#             nn.BatchNorm1d(2),
#             nn.LeakyReLU(0.1),
#             nn.Conv1d(in_channels=2, out_channels=4, kernel_size=[3], stride=1, padding=1),
#             nn.BatchNorm1d(4),
#             nn.LeakyReLU(0.1),
#             nn.AvgPool1d(kernel_size=2, stride=2),
#
#             nn.Conv1d(in_channels=4, out_channels=6, kernel_size=[3], stride=1, padding=1),
#             nn.BatchNorm1d(6),
#             nn.LeakyReLU(0.1),
#             nn.Conv1d(in_channels=6, out_channels=6, kernel_size=[3], stride=1, padding=1),
#             nn.BatchNorm1d(6),
#             nn.LeakyReLU(0.1),
#             nn.AvgPool1d(kernel_size=2, stride=2),
#
#             nn.Conv1d(in_channels=6, out_channels=4, kernel_size=[3], stride=1, padding=1),
#             nn.BatchNorm1d(4),
#             nn.LeakyReLU(0.1),
#             nn.Conv1d(in_channels=4, out_channels=2, kernel_size=[3], stride=1, padding=1),
#             nn.BatchNorm1d(2),
#             nn.LeakyReLU(0.1),
#             nn.AvgPool1d(kernel_size=2, stride=2),
#         )
#
#         self.middle = nn.Sequential(
#             nn.Conv1d(2, 2, kernel_size=[3], stride=1, padding=1),
#             nn.BatchNorm1d(2),
#             nn.LeakyReLU(1)
#         )
#         self.dampLeaner=nn.Sequential(
#             nn.Linear(int((T*SampleRate)/8),64),
#             nn.Tanh(),
#             nn.Linear(64,modelNum)
#         )
#         self.FreqLeaner=nn.Sequential(
#             nn.Linear(int((T * SampleRate) / 8), 64),
#             nn.Tanh(),
#             nn.Linear(64, modelNum)
#         )
#
#         self.dampLearnerDe=nn.Sequential(
#             nn.Linear(modelNum,64),
#             nn.Tanh(),
#             nn.Linear(64,T*SampleRate)
#         )
#
#         self.freqLearnerDe=nn.Sequential(
#             nn.Linear(modelNum,64),
#             nn.Tanh(),
#             nn.Linear(64,T*SampleRate)
#         )
#
#
#         self.decoder = nn.Sequential(
#             nn.Conv1d(in_channels=modelNum*2+1, out_channels=4, kernel_size=[3], stride=1, padding=1),
#             nn.BatchNorm1d(4),
#             nn.LeakyReLU(0.1),
#             nn.Conv1d(in_channels=4, out_channels=6, kernel_size=[3], stride=1, padding=1),
#             nn.BatchNorm1d(6),
#             nn.LeakyReLU(0.1),
#             # nn.Upsample(scale_factor=2),
#
#             nn.Conv1d(in_channels=6, out_channels=6, kernel_size=[3], stride=1, padding=1),
#             nn.BatchNorm1d(6),
#             nn.LeakyReLU(0.1),
#             nn.Conv1d(in_channels=6, out_channels=4, kernel_size=[3], stride=1, padding=1),
#             nn.BatchNorm1d(4),
#             nn.LeakyReLU(0.1),
#             # nn.Upsample(scale_factor=2),
#
#             nn.Conv1d(in_channels=4, out_channels=2, kernel_size=[3], stride=1, padding=1),
#             nn.BatchNorm1d(2),
#             nn.LeakyReLU(0.1),
#             nn.Conv1d(in_channels=2, out_channels=1, kernel_size=[3], stride=1, padding=1),
#             nn.BatchNorm1d(1),
#             nn.LeakyReLU(0.1),
#             # nn.Upsample(scale_factor=2),
#         )
#         # self.endecoder = nn.Sequential(
#         #     nn.Conv1d(in_channels=2, out_channels=4, kernel_size=[3], stride=1, padding=1),
#         #     nn.BatchNorm1d(4),
#         #     nn.Conv1d(in_channels=4, out_channels=6, kernel_size=[3], stride=1, padding=1),
#         #     nn.BatchNorm1d(6),
#         #     nn.AvgPool1d(kernel_size=2, stride=2),
#         #
#         #     nn.Conv1d(in_channels=6, out_channels=8, kernel_size=[3], stride=1, padding=1),
#         #     nn.BatchNorm1d(8),
#         #     nn.Conv1d(in_channels=8, out_channels=10, kernel_size=[3], stride=1, padding=1),
#         #     nn.BatchNorm1d(10),
#         #     nn.AvgPool1d(kernel_size=2, stride=2),
#         #
#         #     nn.Conv1d(in_channels=10, out_channels=8, kernel_size=[3], stride=1, padding=1),
#         #     nn.BatchNorm1d(8),
#         #     nn.Conv1d(in_channels=8, out_channels=6, kernel_size=[3], stride=1, padding=1),
#         #     nn.BatchNorm1d(6),
#         #     nn.Upsample(scale_factor=2),
#         #
#         #     nn.Conv1d(in_channels=6, out_channels=4, kernel_size=[3], stride=1, padding=1),
#         #     nn.BatchNorm1d(4),
#         #     nn.Conv1d(in_channels=4, out_channels=2, kernel_size=[3], stride=1, padding=1),
#         #     nn.BatchNorm1d(2),
#         #
#         #     nn.Conv1d(in_channels=2, out_channels=1, kernel_size=[3], stride=1, padding=1),
#         #     nn.BatchNorm1d(1),
#         #     nn.Upsample(scale_factor=2),
#         # )
#         #
#         # self.output = nn.Sequential(
#         #     nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),
#         #     nn.BatchNorm1d(1),
#         #     nn.LeakyReLU(1),
#         # )
#
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.middle(x)
#         # print(x.shape)
#         x_damp=x[:,0,:]
#         # print(x_damp.shape)
#         x_damp=self.dampLeaner(x_damp)
#         x_dampDecoder=self.dampLearnerDe(x_damp)
#
#         x_Freq=x[:,1,:]
#         x_Freq=self.FreqLeaner(x_Freq)
#         x_FreqDecoder = self.freqLearnerDe(x_Freq)
#
#         x_dampDecoder = x_dampDecoder.view(x_dampDecoder.shape[0], 1, x_dampDecoder.shape[1])
#         x_FreqDecoder = x_FreqDecoder.view(x_FreqDecoder.shape[0], 1, x_FreqDecoder.shape[1])
#
#         t = numpy.linspace(0, T, T * SampleRate)
#         t = dataAbout.numpyTOFloatTensor(t)
#         # t=t.to(device)
#         t=t.to(device)
#         timeStepArr = t.repeat(x_damp.shape[0],1)
#         timeStepArr = timeStepArr.view(x_damp.shape[0],1,timeStepArr.shape[1])
#         # print("x_dampe:",x_dampDecoder.shape,"\tx_freqDecoder:",x_FreqDecoder.shape,"\ttime:",timeStepArr.shape)
#         x_ModelParCon=torch.cat([x_dampDecoder,x_FreqDecoder,timeStepArr],dim=1)
#
#         signal=self.decoder(x_ModelParCon)
#
#         # t = numpy.linspace(0, T, T * SampleRate)
#         # t = dataAbout.numpyTOFloatTensor(t)
#         # # t=t.to(device)
#         # t=t.to(device)
#         # timeStepArr = t.repeat(x_damp.shape[0],1)
#         # timeStepArr = timeStepArr.view(x_damp.shape[0],1,timeStepArr.shape[1])
#         #
#         # signal=0
#         #
#         # for i in range(modelNum):
#         #     dampData = torch.Tensor.exp(torch.mul( (-1)*x_damp[:,i].view(x_damp.shape[0],1,1),timeStepArr))
#         #     freqData = torch.Tensor.sin(torch.mul(2*math.pi*x_Freq[:,i].view(x_Freq.shape[0],1,1),timeStepArr))
#         #
#         #     signal=signal + torch.mul( dampData, freqData)
#
#         # print(signal.shape)
#         return x_damp,x_Freq, signal
#
#
#
#
# ########################################################################################################################
#
# ########################################################################################################################
# class Unet(nn.Module):
#     def __init__(self):
#         super(Unet, self).__init__()
#         print('unet')
#         nlayers = 5
#         nefilters = 2  # 每次迭代时特征增加数量##
#         self.num_layers = nlayers
#         self.nefilters = nefilters
#         filter_size = 3
#         merge_filter_size = 3
#         self.encoder = nn.ModuleList()  # 定义一个空的modulelist命名为encoder##
#         self.decoder = nn.ModuleList()
#         self.ebatch = nn.ModuleList()
#         self.dbatch = nn.ModuleList()
#         echannelin = [1] + [(i + 1) * nefilters for i in range(nlayers - 1)]
#         echannelout = [(i + 1) * nefilters for i in range(nlayers)]
#         self.modelFeatureNum=echannelout[-1]
#         dchannelout = echannelout[::-1]
#         dchannelin = [dchannelout[0] * 2] + [(i) * nefilters + (i - 1) * nefilters for i in range(nlayers, 1, -1)]
#
#         for i in range(self.num_layers):
#             self.encoder.append(
#                 nn.Conv1d(
#                     in_channels=echannelin[i],
#                     out_channels=echannelout[i],
#                     kernel_size=filter_size, stride=1, padding=filter_size // 2
#                 )
#             )
#             self.decoder.append(
#                 nn.Conv1d(
#                     in_channels=dchannelin[i],
#                     out_channels=dchannelout[i],
#                     kernel_size=merge_filter_size, stride=1, padding=merge_filter_size // 2)
#             )
#             self.ebatch.append(nn.BatchNorm1d(echannelout[i]))  # moduleList 的append对象是添加一个层到module中##
#             self.dbatch.append(nn.BatchNorm1d(dchannelout[i]))
#
#         self.middle = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=echannelout[-1],
#                 out_channels=echannelout[-1],
#                 kernel_size=filter_size, stride=1, padding=filter_size // 2
#             ),
#             nn.BatchNorm1d(echannelout[-1]),
#             nn.LeakyReLU(0.1)
#         )
#
#         self.dampLeaner=nn.Sequential(
#             nn.Linear(int((echannelout[-1]/2)*((T*SampleRate)/(2**(nlayers)))),64),
#             nn.Tanh(),
#             nn.Linear(64,modelNum)
#         )
#         self.FreqLeaner=nn.Sequential(
#             nn.Linear(int((echannelout[-1]/2)*((T*SampleRate)/(2**nlayers))), 64),
#             nn.Tanh(),
#             nn.Linear(64, modelNum)
#         )
#
#         self.dampLearnerDe=nn.Sequential(
#             nn.Linear(modelNum,64),
#             nn.Tanh(),
#             nn.Linear(64,int((echannelout[-1]/2)*(T*SampleRate)))
#         )
#
#         self.freqLearnerDe=nn.Sequential(
#             nn.Linear(modelNum,64),
#             nn.Tanh(),
#             nn.Linear(64,int((echannelout[-1]/2)*(T*SampleRate)))
#         )
#
#
#         self.out = nn.Sequential(
#             nn.Conv1d(nefilters + 1, 1, 1),
#             nn.LeakyReLU(1),
#             # nn.Tanh()
#         )
#
#     def forward(self, x):
#         encoder = list()
#         input = x
#
#         for i in range(self.num_layers):
#             x = self.encoder[i](x)
#             x = self.ebatch[i](x)
#             x = F.leaky_relu(x, 0.1)
#             encoder.append(x)
#             # print("x+++",x.shape)
#             x = x[:, :, ::2]
#             # print("x---",x.shape)
#
#         x = self.middle(x)
#
#
#         # print("x:",x.shape)
#         # print("self.modelFeatureNum/2:",self.modelFeatureNum/2)
#         x_damp=x[:,0:int(self.modelFeatureNum/2),:]
#         # print("x_damp:",x_damp.shape)
#         x_damp=x_damp.view(x_damp.shape[0],x_damp.shape[1]*x_damp.shape[2])
#         # print("x_damp",x_damp.shape)
#         x_damp=self.dampLeaner(x_damp)
#         x_dampDecoder=self.dampLearnerDe(x_damp)
#
#         x_Freq=x[:,int(self.modelFeatureNum/2):int(self.modelFeatureNum),:]
#         # print("x_Freq:",x_Freq.shape)
#         x_Freq=x_Freq.view(x_Freq.shape[0],x_Freq.shape[1]*x_Freq.shape[2])
#         x_Freq=self.FreqLeaner(x_Freq)
#         x_FreqDecoder = self.freqLearnerDe(x_Freq)
#
#         x_dampDecoder = x_dampDecoder.view(x_dampDecoder.shape[0], 1, x_dampDecoder.shape[1])
#         x_FreqDecoder = x_FreqDecoder.view(x_FreqDecoder.shape[0], 1, x_FreqDecoder.shape[1])
#         x_ModelParCon=torch.cat([x_dampDecoder,x_FreqDecoder],dim=1)
#
#
#         for i in range(self.num_layers):
#             x = F.interpolate(x, scale_factor=2)
#             x = torch.cat([x, encoder[self.num_layers - i - 1]], dim=1)  ##特征合并过程中维数不对##
#             x = self.decoder[i](x)
#             x = self.dbatch[i](x)
#             x = F.leaky_relu(x, 0.1)
#         x = torch.cat([x, input], dim=1)
#
#         x = self.out(x)
#         return x_damp,x_Freq, x
#
#
# ########################################################################################################################
#
# def train_and_test(NetModel, all_data ,all_label):
#     LOSS_DATA = []
#     LOSS_TEST_DATA1 = []
#     LOSS_TEST_DATA2 = []
#     TRAIN_ACC = []
#     TEST_ACC1 = []
#     TEST_ACC2 = []
#
#     device_ids = [0, 1,2]
#     CNNNet = NetModel().to(device)
#     CNNNet = nn.DataParallel(CNNNet, device_ids=device_ids)
#
#     optimizer = torch.optim.Adam(CNNNet.parameters(), lr=LR)
#     loss_func = nn.MSELoss()
#     # all_label=originalSignalFreq
#
#     for epoch in range(EPOCH):
#
#         train_data, test_data1, train_label, test_label1 \
#             = dataAbout.self_train_test_split(all_data, all_label, TRAIN_TEST_RATE)
#         train_tensor_data = dataAbout.numpyTOFloatTensor(train_data)
#         train_tensor_label = dataAbout.numpyTOFloatTensor(train_label)
#
#
#
#         test_tensor_data1 = dataAbout.numpyTOFloatTensor(test_data1)
#         test_tensor_label1 = dataAbout.numpyTOFloatTensor(test_label1)
#
#
#         train_loadoer = dataAbout.data_loader(train_tensor_data, train_tensor_label)
#         for step, (x, y) in enumerate(train_loadoer):
#             x = Variable(x)
#             # y = Variable(y)
#
#             x = x.to(device)
#             # y = y.to(device)
#
#             damp, freq,output = CNNNet(x)
#
#             timeDomainloss = loss_func(output, x)
#
#             # outputfft, yfft= calculateSpectrumLoss(output, y)
#             # spectrumLoss = loss_func(outputfft, yfft)
#
#             # loss = spectrumLoss+timeDomainloss
#
#             loss = timeDomainloss
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             LOSS_DATA.append(loss.item())
#
#             # testOutput = CNNNet(test_tensor_data1)
#             # lossTest1 = loss_func(testOutput, test_tensor_label1.to(device))
#             # LOSS_TEST_DATA1.append(lossTest1.item())
#
#
#         if epoch % 2 == 0:
#             # print("阻尼：", damp, "\n频率：", freq)
#             print('Epoch: ', epoch,
#                   '| train loss:    ', loss.item(),
#                   # '| test1 loss:    ', lossTest1.item(),
#                   )
#
#         if epoch==EPOCH-1:
#             ResultPathImpulseCondition = mkSaveModelResultdir(
#                 ResultSaveHomePath
#                 + '/impluse' + impulseCondition + '1Order')
#             ResultPathperEPOCH = mkSaveModelResultdir(
#                 ResultPathImpulseCondition + '/E-' + str(EPOCH) + '_LR-' + str(LR)
#                 + '_time-' + str(T) + '_SNR-' + str(SNR))
#             ResultPathModelPath = mkSaveModelResultdir(ResultPathperEPOCH + '/model_result')
#             ResultPathLossPath = mkSaveModelResultdir(ResultPathperEPOCH + '/loss_result')
#             testPredResultCompairPath = mkSaveModelResultdir(
#                 ResultPathperEPOCH +
#                 '/remote_filterResSignal_T-' + str(T) + '_SNR-' + str(SNR) + '_E-' + str(EPOCH) + '_LR-' + str(LR))
#
#
#             for i in range(20):
#                 # print("output:",output[i,:,:].detach().cpu().numpy().T)
#                 plt.figure()
#                 plt.plot(output[i,:,:].detach().cpu().numpy().T)
#                 plt.savefig(
#                     ResultPathLossPath + '/signal_E-%s_LR-%f_Time-%sS_SampleRate-%s-num-%s.png'
#                     % (EPOCH, LR, T, SampleRate,i)
#                 )
#                 plt.close()
#                 # plt.show()
#
#             x_axis = numpy.linspace(1,damp.shape[0], damp.shape[0])
#             plt.figure()
#             plt.scatter(y, damp.detach().cpu().numpy())
#             plt.savefig(
#                 ResultPathLossPath + '/damp_E-%s_LR-%f_Time-%sS_SampleRate-%s.png'
#                 % (EPOCH, LR, T, SampleRate)
#             )
#             plt.close()
#             plt.show()
#             plt.figure()
#             plt.scatter(y, freq.detach().cpu().numpy())
#             plt.savefig(
#                 ResultPathLossPath + '/freq_E-%s_LR-%f_Time-%sS_SampleRate-%s.png'
#                 % (EPOCH, LR, T, SampleRate)
#             )
#             plt.close()
#             plt.show()
#         if epoch == EPOCH + 1:
#
#             ResultPathImpulseCondition = mkSaveModelResultdir(
#                 ResultSaveHomePath
#                 + '/impluse' + impulseCondition +'threeOrder')
#             ResultPathperEPOCH = mkSaveModelResultdir(
#                 ResultPathImpulseCondition + '/E-' + str(EPOCH) + '_LR-' + str(LR)
#                 + '_time-' + str(T) +'_SNR-' + str(SNR) )
#             ResultPathModelPath = mkSaveModelResultdir(ResultPathperEPOCH + '/model_result')
#             ResultPathLossPath = mkSaveModelResultdir(ResultPathperEPOCH + '/loss_result')
#             testPredResultCompairPath = mkSaveModelResultdir(
#                 ResultPathperEPOCH +
#                 '/remote_filterResSignal_T-' + str(T) + '_SNR-' + str(SNR) + '_E-' + str(EPOCH) + '_LR-' + str(LR))
#
#             '''save model'''
#             torch.save(
#                 CNNNet.state_dict(),
#                 ResultPathModelPath + '/model_state_dict_E' + str(epoch + 1) + '_LR-' + str(LR) + '_time-' + str(
#                     T) + '_SNR-' + str(SNR) + '.pth'
#             )
#             torch.save(
#                 CNNNet,
#                 ResultPathModelPath + '/model_NetStructure_E' + str(epoch + 1) + '_LR-' + str(LR) + '_time-' + str(
#                     T) + '_SNR-' + str(SNR) + '.pkl'
#             )
#
#             '''
#             save result
#             '''
#             plt.figure("loss")
#             l1, = plt.plot(LOSS_DATA)
#             l2, = plt.plot(LOSS_TEST_DATA1)
#             l3, = plt.plot(LOSS_TEST_DATA2)
#             plt.xlabel('epoch')
#             plt.ylabel('loss')
#             plt.legend(handles=[l1, l2, l3], labels=['train loss', 'test1 loss', 'test2 loss'], loc='best')
#             plt.title('loss')
#             plt.savefig(
#                 ResultPathLossPath + '/loss_E-%s_LR-%f_Time-%sS_SampleRate-%s.png'
#                 % (EPOCH, LR, T, SampleRate)
#             )
#             plt.savefig(
#                 ResultPathLossPath + '/loss_E-%s_LR-%f_Time-%sS_SampleRate-%s.eps'
#                 % (EPOCH, LR, T, SampleRate)
#             )
#             plt.savefig(
#                 ResultPathLossPath + '/loss_E-%s_LR-%f_Time-%sS_SampleRate-%s.svg'
#                 % (EPOCH, LR, T, SampleRate)
#             )
#             plt.show()
#             lossDataIncludingTrainTest = dataAbout.mergeThreeList(LOSS_DATA, LOSS_TEST_DATA1, LOSS_TEST_DATA2)
#             numpy.savetxt(
#                 ResultPathLossPath + '/loss_E-%s_LR-%f_Time-%sS_SampleRate-%s.txt'
#                 % (EPOCH, LR, T, SampleRate), lossDataIncludingTrainTest
#             )
#
#             '''save test data result'''
#             onestest1 = numpy.ones([int(test_tensor_data1.shape[0] * 0.1), 1])
#             zerostest1 = numpy.zeros([int(test_tensor_data1.shape[0] * 0.9) + 2, 1])
#             boolDatatest1 = numpy.row_stack((onestest1, zerostest1))
#             boolDatatest1 = boolDatatest1.flatten()
#             random.shuffle(boolDatatest1)
#             for i in range(0, test_tensor_data1.shape[0]):
#                 if boolDatatest1[i] == 1:
#                     orignanalResSignal = test_tensor_data1[i, :, :].detach().cpu().numpy().T;
#                     addNoiseResSignal = test_tensor_label1[i, :, :].detach().cpu().numpy().T;
#                     predFilterResSignal = testOutput[i, :, :].detach().cpu().numpy().T;
#                     plt.figure()
#                     plt.subplot(3, 1, 1)
#                     plt.plot(orignanalResSignal)
#                     plt.title('original Response Signal')
#                     plt.subplot(3, 1, 2)
#                     plt.plot(addNoiseResSignal)
#                     plt.title('add Nloise Response Signal')
#                     plt.subplot(3, 1, 3)
#                     plt.plot(predFilterResSignal)
#                     plt.title('prd Filter Response Signal')
#                     plt.savefig(testPredResultCompairPath +
#                                 '/orignal_1Col-addNoise_2col-predFilter_3col-test_Num-'
#                                 + str(i) + '_E-' + str(EPOCH) + '_LR-' + str(LR) + '_timeSeries.png')
#                     plt.close()
#                     testMergedResSysPredSys = numpy.concatenate(
#                         (orignanalResSignal, addNoiseResSignal, predFilterResSignal), axis=1);
#                     numpy.savetxt(
#                         testPredResultCompairPath +
#                         '/orignal_1Col-addNoise_2col-predFilter_3col-test_Num-'
#                         + str(i) + '_E-' + str(EPOCH) + '_LR-' + str(LR) + '.txt', testMergedResSysPredSys)
#
#             # onestest2 = numpy.ones([int(test_tensor_data2.shape[0] * 0.1), 1])
#             # zerostest2 = numpy.zeros([int(test_tensor_data2.shape[0] * 0.9) + 2, 1])
#             # boolDatatest2 = numpy.row_stack((onestest2, zerostest2))
#             # boolDatatest2 = boolDatatest2.flatten()
#             # random.shuffle(boolDatatest2)
#             # for i in range(0, test_tensor_data2.shape[0]):
#             #     if boolDatatest2[i] == 1:
#             #         orignanalResSignal2 = test_tensor_data2[i, :, :].detach().cpu().numpy().T;
#             #         addNoiseResSignal2 = test_tensor_label2[i, :, :].detach().cpu().numpy().T;
#             #         predFilterResSignal2 = testOutput2[i, :, :].detach().cpu().numpy().T;
#             #         plt.figure()
#             #         plt.subplot(3, 1, 1)
#             #         plt.plot(orignanalResSignal2)
#             #         plt.title('original Response Signal')
#             #         plt.subplot(3, 1, 2)
#             #         plt.plot(addNoiseResSignal2)
#             #         plt.title('add Nloise Response Signal')
#             #         plt.subplot(3, 1, 3)
#             #         plt.plot(predFilterResSignal2)
#             #         plt.title('prd Filter Response Signal')
#             #         plt.savefig(independtestPredResultCompairPath +
#             #                     '/orignal_1Col-addNoise_2col-predFilter_3col-independTest_Num-'
#             #                     + str(i) + '_E-' + str(EPOCH) + '_LR-' + str(LR) + '_timeSeries.png')
#             #         plt.close()
#             #         testMergedResSysPredSys2 = numpy.concatenate(
#             #             (orignanalResSignal2, addNoiseResSignal2, predFilterResSignal2), axis=1
#             #         );
#             #         numpy.savetxt(
#             #             independtestPredResultCompairPath +
#             #             '/orignal_1Col-addNoise_2col-predFilter_3col-independTest_Num-'
#             #             + str(i) + '_E-' + str(EPOCH) + '_LR-' + str(LR) + '.txt', testMergedResSysPredSys2)
#
#             # '''
#             # save real data
#             # '''
#             # realDataNameList, realDataOrigianlSiganl = dataAbout.load_real_data(realDataPath)
#             # for signalName, signal in zip(realDataNameList, realDataOrigianlSiganl):
#             #     signal = numpy.reshape(signal, [1, 1, len(signal)])
#             #     signal = (signal - numpy.mean(signal)) / numpy.std(signal)
#             #
#             #     signal = dataAbout.numpyTOFloatTensor(signal)
#             #
#             #     realDataCNNFilterResultTensor = CNNNet(signal)
#             #
#             #     realDataCNNFilterResultTensor = realDataCNNFilterResultTensor
#             #     print(signal.cpu().detach().numpy().flatten())
#             #     print(realDataCNNFilterResultTensor.cpu().detach().numpy())
#             #     realDataOrignalTimeSeries = signal.cpu().detach().numpy().flatten()
#             #     realDataCNNFilterResult = realDataCNNFilterResultTensor.cpu().detach().numpy().flatten()
#             #     realDataOrignalTimeSeriesfft = abs(fft(realDataOrignalTimeSeries))
#             #     realDataCNNFilterResultfft = abs(fft(realDataCNNFilterResult))
#             #
#             #     realDataOrignalAndCNNFilter = dataAbout.mergeTwoList(realDataOrignalTimeSeries.tolist(),
#             #                                                          realDataCNNFilterResult.tolist())
#             #
#             #     realDataResultDicPart = realDataPath.split('/')
#             #     realDataResultDic = realDataResultDicPart[len(realDataResultDicPart) - 1]
#             #
#             #     realDataResultHomePath = mkSaveModelResultdir(ResultPathperEPOCH + '/realDataResult');
#             #     realDataResultsubFolder = mkSaveModelResultdir(
#             #         realDataResultHomePath + '/' + realDataResultDic)
#             #
#             #     numpy.savetxt(realDataResultsubFolder + '/' + signalName[:len(signalName) - 4] + '.txt',
#             #                   realDataOrignalAndCNNFilter)
#             #     print(realDataResultDic)
#             #
#             #     plt.figure()
#             #     plt.subplot(211)
#             #     plt.plot(realDataOrignalTimeSeries)
#             #     plt.title('real Original Signal')
#             #     plt.ylabel('Amplitude')
#             #     plt.subplot(212)
#             #     plt.plot(realDataCNNFilterResult)
#             #     plt.xlabel('len')
#             #     plt.ylabel('Amplitude')
#             #     plt.plot('CNN Filtered Signal')
#             #     plt.savefig(realDataResultsubFolder + '/' + signalName[:len(signalName) - 4] + '_timeSeriesComp.png')
#             #     plt.close()
#             #
#             #     plt.figure()
#             #     plt.subplot(211)
#             #     plt.plot(realDataOrignalTimeSeriesfft)
#             #     plt.title('real Original Signal Frequency Spectrum')
#             #     plt.subplot(212)
#             #     plt.plot(realDataCNNFilterResultfft)
#             #     plt.title('CNN Filtered Signal Frequency Spectrum')
#             #     plt.savefig(
#             #         realDataResultsubFolder + '/' + signalName[:len(signalName) - 4] + 'frequencySpectrumComp.png')
#             #     plt.close()
#             #     # plt.show()
#
# if __name__ == '__main__':
#     EPOCH = 56000
#     BATCH_SIZE = 3000
#
#     LR = 0.0001
#     TRAIN_TEST_RATE = 0.2
#
#     MAXDATASIZE = 3000
#     modelNum=1;
#
# #%%
#     T = 16;
#     SampleRate = 128;
#     SNR = 10;
#     t=numpy.linspace(0,T,T*SampleRate)
#     t=dataAbout.numpyTOFloatTensor(t)
#     # t=t.to(device)
#     signal=numpy.zeros((1,T*SampleRate))
#     signal=dataAbout.numpyTOFloatTensor(signal)
#     signal=signal.to(device)
# #%%
#     # impulseCondition = 'Turbo'
#     impulseCondition = 'pluse'
#
#     # independTestimpulseCondition = 'Turbo'
#     independTestimpulseCondition = 'pluse'
#
#     dataPath = "/media/server/f8d02269-b24b-4f82-9d4d-6e3e4a6fe012/口..口/" \
#                "representationLearning/representationLearningDampFreq/data/" \
#                + impulseCondition + '/1order_'
#     # independTestDataHomePath = "/media/server/f8d02269-b24b-4f82-9d4d-6e3e4a6fe012/" \
#     #                            "口..口/DNNpreprocessing/data/" + independTestimpulseCondition + '/'
#
#     TimeSeriesLength = int(T * SampleRate)
#     path = dataPath + 'DataTimeSeries_' + str(T) + "secondData_SNR-" + str(SNR)
#     originalSignal,originalSignalFreq = dataAbout.load_data(path, TimeSeriesLength)
#
#     # model = flutterFilterNet;
#     model = Unet
#
#     ResultSaveHomePath = "/media/server/f8d02269-b24b-4f82-9d4d-6e3e4a6fe012" \
#                          "/口..口/representationLearning/representationLearningDampFreq/result"
#
#     # independTestDataTimeSample = 16
#     # independSNR = 0
#     # independTestTimeSeriesLength = int(independTestDataTimeSample * SampleRate)
#     # independTestDataPath = \
#     #     independTestDataHomePath + '3order_DataTimeSeries_' + str(independTestDataTimeSample) + \
#     #     "secondData_SNR-" + str(independSNR)
#     # independTestOriginalSignal = dataAbout.load_data(independTestDataPath,
#     #                                                                              independTestTimeSeriesLength)
#     #
#     # realDataPath = \
#     #     '/media/server/f8d02269-b24b-4f82-9d4d-6e3e4a6fe012/口..口/DNNpreprocessing/data/' \
#     #     'Turbo/realdata/all_real_data_zip/all_real_unflutter_last_step_data'
#     # realDataPath = \
#     #     '/media/server/f8d02269-b24b-4f82-9d4d-6e3e4a6fe012/口..口/DNNpreprocessing/data/' \
#     #     'Turbo/realdata/all_real_data_zip/all_real_flutter_last_step_data'
#     # realDataPath = \
#     #     '/media/server/f8d02269-b24b-4f82-9d4d-6e3e4a6fe012/口..口/DNNpreprocessing/data/' \
#     #     'Turbo/realdata/all_real_data_zip/all_real_unflutter_data'
#
#     train_and_test(
#         model, originalSignal,originalSignalFreq)
#
#
'''
表征学习 测试OOM问题
'''

#############################################

##############################################
'''
UNet genTurResSignal OOM tets 
'''

import numpy
import os
from PIL import Image
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt
import locale
import torch.nn.functional as F
import time
from scipy.fftpack import fft


os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class dataAbout():

    def load_data(dataPath, timeSignalLength):
        dataFileNameLists = os.listdir(dataPath);
        excitationSignaDataList = list();
        responseSignalDataList = list();
        timedomainSysFunctionDataList = list();
        # excitationSignaData = numpy.zeros((len(dataFileNameLists),1,timeSignalLength),dtype='float64');
        # responseSignalData = numpy.zeros((len(dataFileNameLists),1,timeSignalLength),dtype='float64');
        # timedomainSysFunctionData = numpy.zeros((len(dataFileNameLists),1,timeSignalLength),dtype='float64');

        for dataFileName in dataFileNameLists:
            dataFilePathName = dataPath + "/" + dataFileName
            allDataFile = open(dataFilePathName)
            allData = numpy.loadtxt(allDataFile)

            pulSignal = allData[:, 0].T
            resSignal = allData[:, 1].T
            timeDomainSys = allData[:, 2].T

            # print("pulSignal", len(pulSignal))
            pulSignal = numpy.reshape(pulSignal, [1, len(pulSignal)])
            resSignal = numpy.reshape(resSignal, [1, len(resSignal)])
            timeDomainSys = numpy.reshape(timeDomainSys, [1, len(timeDomainSys)])
            # plt.figure()
            # plt.plot(pulSignal.flatten())
            # plt.figure()
            # plt.plot(resSignal.flatten())
            # plt.figure()
            # plt.plot(timeDomainSys.flatten())
            # plt.show()
            # plt.close()

            excitationSignaDataList.append(pulSignal)
            responseSignalDataList.append(resSignal)
            timedomainSysFunctionDataList.append(timeDomainSys)
            # print("timedomainSysFunctionDataList", len(timedomainSysFunctionDataList))

            # excitationSignaData[dataFileNameLists.index(dataFileName), :, :] = numpy.reshape(allData[:,0].T,(1,timeSignalLength));
            # responseSignalData[dataFileNameLists.index(dataFileName), :, : ]= numpy.reshape(allData[:,1].T,(1,timeSignalLength));
            # timedomainSysFunctionData[dataFileNameLists.index(dataFileName), :, :] = numpy.reshape(allData[:,2].T,(1,timeSignalLength));

        excitationSignaData = numpy.array(excitationSignaDataList)
        responseSignalData = numpy.array(responseSignalDataList)
        timedomainSysFunctionData = numpy.array(timedomainSysFunctionDataList)

        print("timedomainSysFunctionData", timedomainSysFunctionData.shape)

        return excitationSignaData, responseSignalData, timedomainSysFunctionData

    def self_train_test_split(ALlData, ALlLabel, TRAIN_TEST_RATE):
        TrainData, TestData, TrainLabel, TestLabel \
            = train_test_split(ALlData[:, :, :], ALlLabel[:,:,:], test_size=TRAIN_TEST_RATE,shuffle=False)
        ## 此处MAXDATASIZE 表示读入 的最大数据量
        # = train_test_split(ALlData[:MAXDATASIZE, :, :, :], ALlLabel[:MAXDATASIZE], test_size=TRAIN_TEST_RATE)

        return TrainData, TestData, TrainLabel, TestLabel

    def numpyTOFloatTensor(data):
        data = torch.from_numpy(data)
        tensorData = torch.FloatTensor.float(data)
        return tensorData

    def numpyTOLongTensor(data):
        data = torch.from_numpy(data)
        tensorData = torch.LongTensor.long(data)
        return tensorData

    def modelChoice(model, data1, data2, label):

        if model == 'STFT':
            return data1, label
        if model == 'EMD':
            return data2, label
        if model == 'STFT+EMD':
            data = numpy.concatenate((data1, data2), axis=1)
            return data, label
        else:
            print("(口..口) 没有输入合适的CNN输入模式 (口..口)")
            exit()


    def NetInputLayerNum(model):
        if model == 'STFT':
            return 3
        if model == 'EMD':
            return 1
        if model == 'STFT+EMD':
            return 4
        else:
            print("(口..口) 没有输入合适的CNN输入层的数量 (口..口)")
            exit()

    # 数据封装函数
    def data_loader(data_x, data_y):

        train_data = Data.TensorDataset(data_x, data_y)
        train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
        return train_loader

    def mergeTwoList(List1, List2):

        List1Size = len(List1)
        List2Size = len(List2)

        arrayList1 = numpy.array(List1)
        arrayList1 = arrayList1.reshape(arrayList1.shape[0], arrayList1.shape[1])
        arrayList2 = numpy.array(List2)
        arrayList2 = arrayList2.reshape(arrayList2.shape[0], arrayList2.shape[1])
        mergedArrayList = numpy.concatenate((arrayList1, arrayList2), axis=1)
        return mergedArrayList

    def mergeList(List1, List2, List3):

        List1Size = len(List1)
        List2Size = len(List2)
        List3Size = len(List3)

        arrayList1 = numpy.array(List1)
        arrayList1 = arrayList1.reshape(List1Size, 1)
        arrayList2 = numpy.array(List2)
        arrayList2 = arrayList2.reshape(List2Size, 1)
        arrayList3 = numpy.array(List3)
        arrayList3 = arrayList3.reshape(List3Size, 1)
        mergedArrayList = numpy.concatenate((arrayList1, arrayList2, arrayList3), axis=1)
        return mergedArrayList


class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        print('unet')
        nlayers = LayerNumber
        nefilters=NumberofFeatureChannel # 每次迭代时特征增加数量###
        self.num_layers = nlayers
        self.nefilters = nefilters
        filter_size = 11
        merge_filter_size = 11
        self.encoder = nn.ModuleList() # 定义一个空的modulelist命名为encoder#####
        self.decoder = nn.ModuleList()
        self.ebatch = nn.ModuleList()
        self.dbatch = nn.ModuleList()
        echannelin = [1] + [(i + 1) * nefilters for i in range(nlayers-1)]
        echannelout = [(i + 1) * nefilters for i in range(nlayers)]
        dchannelout = echannelout[::-1]
        dchannelin = [dchannelout[0]*2]+[(i) * nefilters + (i - 1) * nefilters for i in range(nlayers,1,-1)]

        for i in range(self.num_layers):
            self.encoder.append(nn.Conv1d(echannelin[i],echannelout[i],filter_size,padding=filter_size//2))
            self.decoder.append(nn.Conv1d(dchannelin[i],dchannelout[i],merge_filter_size,padding=merge_filter_size//2))
            self.ebatch.append(nn.BatchNorm1d(echannelout[i]))  #  moduleList 的append对象是添加一个层到module中######
            self.dbatch.append(nn.BatchNorm1d(dchannelout[i]))

        self.middle = nn.Sequential(
            nn.Conv1d(echannelout[-1],echannelout[-1],filter_size,padding=filter_size//2),
            nn.BatchNorm1d(echannelout[-1]),
            nn.LeakyReLU(0.1)
        )
        self.out = nn.Sequential(
            #
            nn.Conv1d(nefilters + 1, 1, 1),
            nn.Tanh(),
        )
        self.lstmlayer= nn.LSTM(
                input_size=T*Fs,
                hidden_size=T*Fs,
                num_layers=1,
                batch_first=True
        )
    def forward(self,x):

        encoder = list()
        input = x
        # print(x.shape)


        for i in range(self.num_layers):
            x = self.encoder[i](x)
            x = self.ebatch[i](x)
            x = F.leaky_relu(x,0.1)
            encoder.append(x)
            x = x[:,:,::2]
            # print(x.shape)

        x = self.middle(x)

        for i in range(self.num_layers):
            x = F.upsample(x,scale_factor=2,mode='linear')
            if(i<self.num_layers/2):
                x = torch.cat([x,encoder[self.num_layers - i - 1]],dim=1)  ##特征合并过程中维数不对#######
            else:
                x = torch.cat([x, encoder[self.num_layers - i - 1]], dim=1)  ##特征合并过程中维数不对#######
            x = self.decoder[i](x)
            x = self.dbatch[i](x)
            x = F.leaky_relu(x,0.1)
        x = torch.cat([x,input],dim=1)

        x = self.out(x)

        # x,(h_n,h_c)=self.lstmlayer(x,(None))
        return x

def train_and_test(NetModel,all_data,all_label):
    LOSS_DATA = []
    LOSS_DATA_timeSeries=[]
    LOSS_DATA_spectrum=[]
    LOSS_TEST_DATA1 = []
    LOSS_TEST_timeSeries=[]
    LOSS_TEST_spectrun=[]
    TRAIN_ACC = []
    TEST_ACC1 = []

    device_ids = [0, 1, 2]
    CNNNet = NetModel().to(device)
    CNNNet = nn.DataParallel(CNNNet, device_ids=device_ids)

    optimizer = torch.optim.Adam(CNNNet.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.MSELoss()

    for epoch in range(EPOCH):

        train_data, test_data1, train_label, test_label1 \
            = dataAbout.self_train_test_split(all_data, all_label, TRAIN_TEST_RATE)
        train_tensor_data = dataAbout.numpyTOFloatTensor(train_data)
        train_tensor_label = dataAbout.numpyTOFloatTensor(train_label)

        test_tensor_data1 = dataAbout.numpyTOFloatTensor(test_data1)
        test_tensor_label1 = dataAbout.numpyTOFloatTensor(test_label1)

        # print(train_tensor_label.dtype)

        train_loadoer = dataAbout.data_loader(train_tensor_data, train_tensor_label)
        for step, (x, y) in enumerate(train_loadoer):
            x = Variable(x)
            y = Variable(y)

            x = x.to(device)
            y = y.to(device)
            # print(y.dtype)

            # with torch.no_grad():

            output = CNNNet(x)

            # print(output.shape)
            # print(y.shape)

            '''train data time loss'''
            timeSeriesloss = loss_func(output[:,:,:int(output.shape[2]/lossRate)], y[:,:,:int(y.shape[2]/lossRate)])

            '''train data stft loss'''
            ystft=torch.stft(y.view(y.shape[0],y.shape[2])[:,:int(y.shape[2]/lossRate)],Fs)
            outputstft=torch.stft(output.view(output.shape[0],output.shape[2])[:,:int(output.shape[2]/lossRate)],Fs)
            spectrumLoss=loss_func(ystft,outputstft)
            '''train data all loss'''
            loss=0.5*timeSeriesloss+0.5*spectrumLoss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            LOSS_DATA.append(loss.item())
            LOSS_DATA_timeSeries.append(timeSeriesloss.item())
            LOSS_DATA_spectrum.append(spectrumLoss.item())

            testOutput = CNNNet(test_tensor_data1)

            test_tensor_label1_GPU= test_tensor_label1.to(device)
            lossTest1timeSeries = loss_func(testOutput[:, :, :int(testOutput.shape[2]/lossRate)],
                                            test_tensor_label1_GPU[:, :,  int(test_tensor_label1_GPU.shape[2] / 4)])
            '''test data time loss'''

            '''test data stft loss'''
            testOutputSTFT = torch.stft(
                testOutput.view(testOutput.shape[0], testOutput.shape[2])[:,:int(testOutput.shape[2] / 4)], Fs)
            testLabel1STFT = torch.stft(
                test_tensor_label1_GPU.view(test_tensor_label1_GPU.shape[0], test_tensor_label1_GPU.shape[2])[:,:
                int(test_tensor_label1_GPU.shape[2] / 4)], Fs)

            lossTest1Spectrum=loss_func(testLabel1STFT,testOutputSTFT)
            '''test data all loss'''
            lossTest1=0.5*lossTest1timeSeries+0.5*lossTest1Spectrum

            LOSS_TEST_DATA1.append(lossTest1.item())
            LOSS_TEST_timeSeries.append(lossTest1timeSeries.item())
            LOSS_TEST_spectrun.append(lossTest1Spectrum.item())


        if epoch % 2 == 0:
            # print('hahaha', numpy.sum(test_acc == 0))
            print('Epoch: ', epoch,
                  '| train loss:  ', loss.item(),"\t",spectrumLoss.item(),"\t",timeSeriesloss.item(),
                  '| test1 loss: ', lossTest1.item(),"\t",lossTest1Spectrum.item(),"\t",lossTest1timeSeries.item()
                      )
        if epoch == EPOCH - 1:

            curTrainResultSaveHomePath = mkSaveModelResultdir(
                ResultSaveHomePath+
                '/turbhtGen_1order__E-'+str(EPOCH)+"_LR-"+str(LR)+'_LayerNum-'+str(LayerNumber)+'_filterNum-'+str(NumberofFeatureChannel))

            '''save loss'''
            resultLossDataPath = mkSaveModelResultdir(curTrainResultSaveHomePath+'/loss_result')
            LOSS_DATA_allMerge = dataAbout.mergeList(LOSS_DATA,LOSS_DATA_timeSeries,LOSS_DATA_spectrum)
            LOSS_TEST_allMerge = dataAbout.mergeList(LOSS_TEST_DATA1,LOSS_TEST_timeSeries,LOSS_TEST_spectrun)
            LOSS_mergedTrainTestData = dataAbout.mergeTwoList(LOSS_DATA_allMerge,LOSS_TEST_allMerge)

            plt.figure("loss time&spectrum")
            l1, = plt.plot(LOSS_DATA)
            l2, = plt.plot(LOSS_TEST_DATA1)
            plt.xlabel('epoch')
            plt.ylabel('loss time&spectrum')
            plt.legend(handles = [l1, l2],labels = ['train loss', 'test loss'],loc = 'best')
            plt.title('loss time&spectrum')
            plt.savefig(
                resultLossDataPath+'/loss_time&spectrum_E-%s_LR-%f_Time-%sS.png'
                %(EPOCH, LR, T)
            )
            plt.savefig(
                resultLossDataPath + '/loss_time&spectrum_E-%s_LR-%f_Time-%sS.eps'
                % (EPOCH, LR, T)
            )
            plt.savefig(
                resultLossDataPath + '/loss_time&spectrum_E-%s_LR-%f_Time-%sS.svg'
                % (EPOCH, LR, T)
            )
            plt.close()

            plt.figure("loss time")
            l1, = plt.plot(LOSS_DATA_timeSeries)
            l2, = plt.plot(LOSS_TEST_timeSeries)
            plt.xlabel('epoch')
            plt.ylabel('loss time')
            plt.legend(handles = [l1, l2],labels = ['train time loss', 'test time loss'],loc = 'best')
            plt.title('loss time')
            plt.savefig(
                resultLossDataPath+'/loss_time_E-%s_LR-%f_Time-%sS.png'
                %(EPOCH, LR, T)
            )
            plt.savefig(
                resultLossDataPath + '/loss_time_E-%s_LR-%f_Time-%sS.eps'
                % (EPOCH, LR, T)
            )
            plt.savefig(
                resultLossDataPath + '/loss_time_E-%s_LR-%f_Time-%sS.svg'
                % (EPOCH, LR, T)
            )
            plt.close()

            plt.figure("loss spectrum")
            l1, = plt.plot(LOSS_DATA_timeSeries)
            l2, = plt.plot(LOSS_TEST_timeSeries)
            plt.xlabel('epoch')
            plt.ylabel('loss spectrum')
            plt.legend(handles = [l1, l2],labels = ['train spectrum loss', 'test spectrum loss'],loc = 'best')
            plt.title('loss time')
            plt.savefig(
                resultLossDataPath+'/loss_spectrum_E-%s_LR-%f_Time-%sS.png'
                %(EPOCH, LR, T)
            )
            plt.savefig(
                resultLossDataPath + '/loss_spectrum_E-%s_LR-%f_Time-%sS.eps'
                % (EPOCH, LR, T)
            )
            plt.savefig(
                resultLossDataPath + '/loss_spectrum_E-%s_LR-%f_Time-%sS.svg'
                % (EPOCH, LR, T)
            )
            plt.close()

            numpy.savetxt(
                resultLossDataPath + '/loss_train&Test_all&time&spectrum_E-%s_LR-%f_Time-%sS.txt'
                % (EPOCH, LR, T),LOSS_mergedTrainTestData
            )

            '''save model'''
            ResultPathModelPath = mkSaveModelResultdir(curTrainResultSaveHomePath+"/model_result")
            torch.save(
                CNNNet.state_dict(),
                ResultPathModelPath +
                '/model_state_dict_E' + str(T) + '_Fs-' + str(Fs) + '_LayerNum-' + str(LayerNumber) + '_filterNum-' +
                str(NumberofFeatureChannel) + '_Epoch-' + str(EPOCH) + '_LR-' + str(LR)+'.pth'
            )
            torch.save(
                CNNNet,
                ResultPathModelPath +
                '/model_NetStructure_E' + str(T) + '_Fs-' + str(Fs) + '_LayerNum-' + str(LayerNumber) + '_filterNum-' +
                str(NumberofFeatureChannel) + '_Epoch-' + str(EPOCH) + '_LR-' + str(LR)+ '.pkl'
            )


            '''测试结果 '''
            testResultSavePath = mkSaveModelResultdir(curTrainResultSaveHomePath+'/testData_result')
            testPredResultCompairPath = mkSaveModelResultdir(
                testResultSavePath +
                '/remoteHost_testResSys_T-' + str(T) + '_Fs-' + str(Fs) + '_LayerNum-' + str(
                    LayerNumber) + '_filterNum-' +
                str(NumberofFeatureChannel) + '_Epoch-' + str(EPOCH) + '_LR-' + str(LR))

            for i in range(0,test_tensor_data1.shape[0]):
                if i % 5 == 0:
                    testResponseSignal = test_tensor_data1[i,:,:].detach().cpu().numpy().T;
                    testTimeSys = test_tensor_label1[i,:,:].detach().cpu().numpy().T;
                    predTestTime = testOutput[i,:,:].detach().cpu().numpy().T;
                    testMergedResSysPredSys = numpy.concatenate((testResponseSignal,testTimeSys,predTestTime),axis=1);

                    numpy.savetxt(
                    testPredResultCompairPath+
                    '/test_1Col-Res_2col-Sys_3col-predSys_Num-'+str(i)+'_EPOCH-'+str(EPOCH)+'_LR-'+str(LR)+'.txt',testMergedResSysPredSys)


                    # plt.figure()
                    # plt.subplot(3,1,1)
                    # plt.plot(testResponseSignal)
                    # plt.title("turResSignia")
                    # plt.subplot(3,1,2)
                    # plt.plot(testTimeSys)
                    # plt.title("turSys")
                    # plt.subplot(3,1,3)
                    # plt.plot(predTestTime)
                    # plt.title("predTurSys")
                    # plt.savefig(
                    #         testPredResultCompairPath +
                    #         '/test_1Col-Res_2col-Sys_3col-predSys_Num-'+str(i)+'_EPOCH-'+str(EPOCH)+'_LR-'+str(LR)+'.svg',dpi=600,format='svg')
                    # plt.savefig(
                    #         testPredResultCompairPath +
                    #         '/test_1Col-Res_2col-Sys_3col-predSys_Num-'+str(i)+'_EPOCH-'+str(EPOCH)+'_LR-'+str(LR)+'.eps',dpi=600,format='eps')
                    # plt.close()
                    #
                    # plt.figure()
                    # plt.subplot(3,1,1)
                    # plt.plot(abs(fft(testResponseSignal)))
                    # plt.title("turResSigniaFFT")
                    # plt.subplot(3,1,2)
                    # plt.plot(abs(fft(testTimeSys)))
                    # plt.title("turSysFFT")
                    # plt.subplot(3,1,3)
                    # plt.plot(abs(fft(predTestTime)))
                    # plt.title("predTurSysFFT")
                    # plt.savefig(
                    #         testPredResultCompairPath +
                    #         '/testFFT_1Col-Res_2col-Sys_3col-predSys_Num-'+str(i)+'_EPOCH-'+str(EPOCH)+'_LR-'+str(LR)+'.svg',dpi=600,format='svg')
                    # plt.savefig(
                    #         testPredResultCompairPath +
                    #         '/testFFT_1Col-Res_2col-Sys_3col-predSys_Num-'+str(i)+'_EPOCH-'+str(EPOCH)+'_LR-'+str(LR)+'.eps',dpi=600,format='eps')
                    # plt.close()



            # resultShowandSave(
            #     CNNNet, LOSS_DATA, LOSS_TEST_DATA1,  TRAIN_ACC, TEST_ACC1, modelHomePath,
            #     ResultHomePath)

        # if epoch == EPOCH - 1:
        #     '''输出结果-数据折叠 '''
        #     resultShowandSave(
        #         CNNNet, LOSS_DATA, LOSS_TEST_DATA1, LOSS_TEST_DATA2, TRAIN_ACC, TEST_ACC1, TEST_ACC2, modelHomePath,
        #         ResultHomePath)
def mkSaveModelResultdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        return path
    else:
        return path

def pathSplit(dataPath):
    splitedDataPath = dataPath.split('_')
    timeStr = splitedDataPath[1];
    FsStr = splitedDataPath[2];
    timeNumStrincludeS = timeStr.split('-')[1]
    timeNumStr = timeNumStrincludeS[:len(timeNumStrincludeS) - 1]
    timeNum = int(timeNumStr)
    FsNumStr = FsStr.split('-')[1]
    FsNum = int(FsNumStr)
    return timeNum,FsNum
def main(path):
    # path = '/home/server/Duanshiqiang/DLGenerateFRFHwHt/data/turbulentExcitation_T-4s_Fs-512';
    excitation, response,timeSys = dataAbout.load_data(path,T*Fs);
    model = Unet;
    train_and_test(model, response, timeSys)


if __name__=='__main__':

    path = '/media/server/f8d02269-b24b-4f82-9d4d-6e3e4a6fe012/口..口/Duanshiqiang/' \
           'DLGenerateFRFHwHt/data/tur_T-20s_Fs-512';
    ResultSaveHomePath = '/media/server/f8d02269-b24b-4f82-9d4d-6e3e4a6fe012/口..口/Duanshiqiang/DLGenerateFRFHwHt/result';
    MAXDATASIZE = 3000;
    TRAIN_TEST_RATE = 0.2;
    BATCH_SIZE = int((MAXDATASIZE*(1-TRAIN_TEST_RATE)));
    # BATCH_SIZE=100
    time,sampleRate = pathSplit(path)
    T = time;
    Fs = sampleRate;

    lossRate = 1

    LayerNumber = 5
    NumberofFeatureChannel = 2
    timeLength = T*Fs;
    EPOCH = 50001;
    LR = 0.0001;

    ''' loss 取1/4即可'''

    main(path)



