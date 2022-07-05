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

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class dataAbout():
    def load_data(dataPath,timeSignalLength):
        dataFileNameLists = os.listdir(dataPath);
        excitationSignaData = numpy.empty((len(dataFileNameLists),1,timeSignalLength),dtype='float64');
        responseSignalData = numpy.empty((len(dataFileNameLists),1,timeSignalLength),dtype='float64');
        timedomainSysFunctionData = numpy.empty((len(dataFileNameLists),1,timeSignalLength),dtype='float64');

        for dataFileName in dataFileNameLists:
            dataFilePathName = dataPath+"/"+dataFileName
            allDataFile = open(dataFilePathName)
            allData = numpy.loadtxt(allDataFile)

            excitationSignaData[dataFileNameLists.index(dataFileName), :, :] = numpy.reshape(allData[:,0].T,(1,timeSignalLength));
            responseSignalData[dataFileNameLists.index(dataFileName), :, : ]= numpy.reshape(allData[:,1].T,(1,timeSignalLength));
            timedomainSysFunctionData[dataFileNameLists.index(dataFileName), :, :] = numpy.reshape(allData[:,2].T,(1,timeSignalLength));

        return excitationSignaData, responseSignalData, timedomainSysFunctionData

    def self_train_test_split(ALlData, ALlLabel, TRAIN_TEST_RATE):
        TrainData, TestData, TrainLabel, TestLabel \
            = train_test_split(ALlData[:MAXDATASIZE, :, :], ALlLabel[:MAXDATASIZE,:,:], test_size=TRAIN_TEST_RATE,shuffle=False)
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
        nefilters=NumberofFeatureChannel # 每次迭代时特征增加数量
        self.num_layers = nlayers
        self.nefilters = nefilters
        filter_size = 5
        merge_filter_size = 5
        self.encoder = nn.ModuleList() # 定义一个空的modulelist命名为encoder
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
            self.ebatch.append(nn.BatchNorm1d(echannelout[i]))  #  moduleList 的append对象是添加一个层到module中
            self.dbatch.append(nn.BatchNorm1d(dchannelout[i]))

        self.middle = nn.Sequential(
            nn.Conv1d(echannelout[-1],echannelout[-1],filter_size,padding=filter_size//2), # //双斜杠取整
            nn.BatchNorm1d(echannelout[-1]),
            nn.LeakyReLU(0.1)
        )
        self.out = nn.Sequential(
            nn.Conv1d(nefilters + 1, 1, 1),
            nn.Tanh()
        )
    def forward(self,x):
        encoder = list()
        input = x


        for i in range(self.num_layers):
            x = self.encoder[i](x)
            x = self.ebatch[i](x)
            x = F.leaky_relu(x,0.1)
            encoder.append(x)
            x = x[:,:,::2]
            # print(x.shape)

        x = self.middle(x)
        # print(x.shape)

        for i in range(self.num_layers):
            x = F.upsample(x,scale_factor=2,mode='linear')
            # print('deocder_dim：',x.shape,
            #       '\tencode_dim:',encoder[self.num_layers - i - 1].shape)
            x = torch.cat([x,encoder[self.num_layers - i - 1]],dim=1)  ##特征合并过程中维数不对
            x = self.decoder[i](x)
            x = self.dbatch[i](x)
            x = F.leaky_relu(x,0.1)
        x = torch.cat([x,input],dim=1)

        x = self.out(x)

        return x

def train_and_test(NetModel,all_data,all_label):
    LOSS_DATA = []
    LOSS_TEST_DATA1 = []
    LOSS_TEST_DATA2 = []
    TRAIN_ACC = []
    TEST_ACC1 = []
    TEST_ACC2 = []

    device_ids = [0, 1,2]
    CNNNet = NetModel().to(device)
    CNNNet = nn.DataParallel(CNNNet, device_ids=device_ids)

    optimizer = torch.optim.Adam(CNNNet.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.MSELoss()  # the target label is not one-hotted

    for epoch in range(EPOCH):

        train_data, test_data1, train_label, test_label1 \
            = dataAbout.self_train_test_split(all_data, all_label, TRAIN_TEST_RATE)
        train_tensor_data = dataAbout.numpyTOFloatTensor(train_data)
        train_tensor_label = dataAbout.numpyTOFloatTensor(train_label)

        test_tensor_data1 = dataAbout.numpyTOFloatTensor(test_data1)
        test_tensor_label1 = dataAbout.numpyTOFloatTensor(test_label1)

        train_loadoer = dataAbout.data_loader(train_tensor_data, train_tensor_label)
        for step, (x, y) in enumerate(train_loadoer):
            x = Variable(x)
            y = Variable(y)

            x = x.to(device)
            y = y.to(device)

            output = CNNNet(x)

            loss = loss_func(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            LOSS_DATA.append(loss.item())

            testOutput = CNNNet(test_tensor_data1)
            lossTest1 = loss_func(testOutput, test_tensor_label1.to(device))
            LOSS_TEST_DATA1.append(lossTest1.item())

        if epoch % 2 == 0:
            # print('hahaha', numpy.sum(test_acc == 0))
            print('Epoch: ', epoch,
                  '| train loss:  ', loss.item(),
                  '| test1 loss: ', lossTest1.item(),
                     )

        if epoch == EPOCH - 1:
            '''输出结果-数据折叠 '''
            for i in range(0,test_tensor_data1.shape[0]):
                testResponseSignal = test_tensor_data1[i,:,:].detach().cpu().numpy().T;
                # print(test_tensor_data1[i,:,:].numpy().shape)
                testTimeSys = test_tensor_label1[i,:,:].detach().cpu().numpy().T;
                predTestTime = testOutput[i,:,:].detach().cpu().numpy().T;
                testMergedResSysPredSys = numpy.concatenate((testResponseSignal,testTimeSys,predTestTime),axis=1);
                testPredResultCompairPath = mkSaveModelResultdir(
                    ResultSaveHomePath+
                    '/loacal_testResSys_LayerNum-'+str(LayerNumber)+'_filterNum'+str(NumberofFeatureChannel)+
                    '_Epoch-'+str(EPOCH)+'_LR-'+str(LR))
                numpy.savetxt(
                    testPredResultCompairPath+
                    '/test_1Col-Res_2col-Sys_3col-predSys_Num-'+str(i)+'_EPOCH-'+str(EPOCH)+'_LR-'+str(LR)+'.txt',testMergedResSysPredSys)

            # resultShowandSave(
            #     CNNNet, LOSS_DATA, LOSS_TEST_DATA1,  TRAIN_ACC, TEST_ACC1, modelHomePath,
            #     ResultHomePath)

def mkSaveModelResultdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        return path
    else:
        return path

#
# def resultShowandSave(
#         Net,lossTrainData,lossTestData1,AccTrainData,AccTestData1,modelHomePath,ResultHomePath):
#
#     constantModelHomePath = mkSaveModelResultdir(
#         modelHomePath + '/svgResult/' + CNNConstructure +  CNNInputMODEL+'_'+ResultLable+'_Time-'+str(T))
#     constantResultHomePath = mkSaveModelResultdir(
#         ResultHomePath + '/svgResult/' + CNNConstructure +  CNNInputMODEL+'_'+ResultLable+'_Time-'+str(T))
#
#     torch.save(
#         Net,
#         constantModelHomePath +
#         '/model_%s_EPOCH-%s_LR-%f_Time-%ss_MDATA-%s_CNNInputMODEL-%s.pkl'
#         % (ComparisonModel, EPOCH, LR, T,  MAXDATASIZE,  CNNInputMODEL))
#
#     plt.figure('loss')
#     l1, = plt.plot(lossTrainData, 'r')
#     l2, = plt.plot(lossTestData1, 'b--')
#     plt.ylim(-0.01, 1.8)
#     plt.xlabel('epoch')
#     plt.ylabel('loss')
#     plt.legend(handles=[l1, l2,l3], labels=['train loss', 'test1 loss','test2 loss'], loc='best')
#     plt.title('loss')
#     plt.savefig(
#         constantResultHomePath +
#         '/loss_%s_EPOCH-%s_LR-%f_Time-%s_MDATA-%s_CNNInputMODEL-%s.svg'
#         % (ComparisonModel, EPOCH, LR, T,  MAXDATASIZE,  CNNInputMODEL),dpi=600,format='svg')
#     plt.savefig(
#         constantResultHomePath +
#         '/loss_%s_EPOCH-%s_LR-%f_Time-%ss_MDATA-%s_NNInputMODEL-%s.eps'
#         % (ComparisonModel, EPOCH, LR, T,  MAXDATASIZE,  CNNInputMODEL),dpi=600,format='eps')
#
#     lossResultincludeTrainTest1Test2 = dataAbout.mergeList(lossTrainData,lossTestData1,lossTestData2)
#     numpy.savetxt(
#         constantResultHomePath +
#         '/ loss_%s_EPOCH-%s_LR-%f_Time-%ss_MDATA-%s_CNNInputMODEL-%s.txt'
#         % (ComparisonModel, EPOCH, LR, T, MAXDATASIZE,  CNNInputMODEL), lossResultincludeTrainTest1Test2)
#
#
#     plt.figure('accuracy')
#     l11, = plt.plot(AccTrainData, 'r')
#     l12, = plt.plot(AccTestData1, 'b--')
#     plt.ylim(0, 1.1)
#     plt.xlabel('epoch')
#     plt.ylabel('accuracy')
#     plt.legend(handles=[ l11,l12,l13], labels=['train acc','test1 acc','test2 acc'], loc='best')
#     plt.title('accuracy rate')
#     plt.savefig(
#         constantResultHomePath +
#         '/acc_%s_EPOCH-%s_LR-%f_Time-%ss_MAXDATASIZE-%s_CNNInputMODEL-%s.svg'
#         % (ComparisonModel, EPOCH, LR, T,  MAXDATASIZE,  CNNInputMODEL),dpi=600,format='svg')
#     plt.savefig(
#         constantResultHomePath +
#         '/acc_%s_EPOCH-%s_LR-%f_Time-%ss_MDATA-%s_CNNInputMODEL-%s.eps'
#         % (ComparisonModel, EPOCH, LR, T,  MAXDATASIZE,  CNNInputMODEL),dpi=600,format='eps')
#     AccResultincludeTrainTest1Test2 = dataAbout.mergeList(AccTrainData, AccTestData1, AccTestData2)
#     numpy.savetxt(
#         constantResultHomePath +
#         '/acc_%s_EPOCH-%s_LR-%f_Time-%ss_MDATA-%s_CNNInputMODEL-%s.txt'
#         % (ComparisonModel, EPOCH, LR, T,  MAXDATASIZE,  CNNInputMODEL), AccResultincludeTrainTest1Test2)
#
#     plt.show()




def main():

    excitation, response,timeSys = dataAbout.load_data(path,T*Fs);
    model = Unet;
    train_and_test(model, response, timeSys)


if __name__=='__main__':
    path = '/media/server/f8d02269-b24b-4f82-9d4d-6e3e4a6fe012/口..口/Duanshiqiang/' \
           'DLGenerateFRFHwHt/data/turbulentExcitation_T-20s_Fs-512';
    ResultSaveHomePath = '/media/server/f8d02269-b24b-4f82-9d4d-6e3e4a6fe012/口..口/Duanshiqiang/DLGenerateFRFHwHt/result';

    MAXDATASIZE = 3000;
    TRAIN_TEST_RATE = 0.2;

    BATCH_SIZE = int((MAXDATASIZE*(1-TRAIN_TEST_RATE)));
    T=20;
    Fs=512;
    LayerNumber = 10;
    NumberofFeatureChannel = 2;
    timeLength = T*Fs;
    EPOCH=50000;
    LR = 0.0001;

    main()



