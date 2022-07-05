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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class dataAbout():
    def load_data(dataPath,timeSignalLength):
        dataFileNameLists = os.listdir(dataPath);
        excitationSignaData = numpy.empty((len(dataFileNameLists),1,timeSignalLength),dtype='float32');
        responseSignalData = numpy.empty((len(dataFileNameLists),1,timeSignalLength),dtype='float32');
        timedomainSysFunctionData = numpy.empty((len(dataFileNameLists),1,timeSignalLength),dtype='float32');

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



class flutterSysGenNet(nn.Module):
    def __init__(self):
        super(flutterSysGenNet, self).__init__()
        filterKernelSize=5
        # self.filterKernelSize=filterKernelSize
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=filterKernelSize, stride=1, padding=filterKernelSize//2),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(in_channels=2, out_channels=4, kernel_size=filterKernelSize, stride=1, padding=filterKernelSize//2),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.1),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=4, out_channels=6, kernel_size=filterKernelSize, stride=1, padding=filterKernelSize//2),
            nn.BatchNorm1d(6),
            nn.LeakyReLU(0.1),
            nn.Conv1d(in_channels=6, out_channels=8, kernel_size=filterKernelSize, stride=1, padding=filterKernelSize//2),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.1),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=8, out_channels=10, kernel_size=filterKernelSize, stride=1, padding=filterKernelSize//2),
            nn.BatchNorm1d(10),
            nn.LeakyReLU(0.1),
            nn.Conv1d(in_channels=10, out_channels=12, kernel_size=filterKernelSize, stride=1, padding=filterKernelSize//2),
            nn.BatchNorm1d(12),
            nn.LeakyReLU(0.1),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )

        self.middle = nn.Sequential(
            nn.Conv1d(12, 12, kernel_size=filterKernelSize, stride=1, padding=filterKernelSize//2),
            nn.BatchNorm1d(12),
            # nn.LeakyReLU(0.1)
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=10, kernel_size=filterKernelSize, stride=1, padding=filterKernelSize//2),
            nn.BatchNorm1d(10),
            nn.LeakyReLU(0.1),
            nn.Conv1d(in_channels=10, out_channels=8, kernel_size=filterKernelSize, stride=1, padding=filterKernelSize//2),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(in_channels=8, out_channels=6, kernel_size=filterKernelSize, stride=1, padding=filterKernelSize//2),
            nn.BatchNorm1d(6),
            nn.LeakyReLU(0.1),
            nn.Conv1d(in_channels=6, out_channels=4, kernel_size=filterKernelSize, stride=1, padding=filterKernelSize//2),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(in_channels=4, out_channels=2, kernel_size=filterKernelSize, stride=1, padding=filterKernelSize//2),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(in_channels=2, out_channels=1, kernel_size=filterKernelSize, stride=1, padding=filterKernelSize//2),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2),
        )


        self.output = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=1,kernel_size=1, stride=1,padding=0),
            # nn.LeakyReLU(1)
            nn.Tanh()
        )


    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = self.output(x)

        return x
######################################

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
            nn.Conv1d(nefilters + 1, 1, 1),
            nn.Tanh()
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
            # print('deconder_dim:',x.shape,
            #       'encoder_dim:',encoder[self.num_layers - i - 1].shape)####
            x = torch.cat([x,encoder[self.num_layers - i - 1]],dim=1)  ##特征合并过程中维数不对#######
            x = self.decoder[i](x)
            x = self.dbatch[i](x)
            x = F.leaky_relu(x,0.1)
        x = torch.cat([x,input],dim=1)

        x = self.out(x)
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

            output = CNNNet(x)

            # print(output.dtype)
            # print(y.dtype)

            '''train data time loss'''
            timeSeriesloss = loss_func(output, y)

            '''train data stft loss'''
            ystft=torch.stft(y.view(y.shape[0],y.shape[2]),Fs)
            outputstft=torch.stft(output.view(output.shape[0],output.shape[2]),Fs)
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
            '''test data time loss'''
            lossTest1timeSeries = loss_func(testOutput, test_tensor_label1.to(device))
            '''test data stft loss'''
            test_tensor_label1_GPU= test_tensor_label1.to(device)
            testLabel1STFT=torch.stft(test_tensor_label1_GPU.view(test_tensor_label1_GPU.shape[0],test_tensor_label1_GPU.shape[2]),Fs)
            testOutputSTFT=torch.stft(testOutput.view(testOutput.shape[0],testOutput.shape[2]),Fs)
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
                '/turbhtGen_flutterSysGenNet_E-'+
                str(EPOCH)+"_LR-"+str(LR)+'_LayerNum-'+str(LayerNumber)+'_filterNum-'+str(NumberofFeatureChannel))

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
                if i % 2 == 0:
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
    excitation, response, timeSys = dataAbout.load_data(path,T*Fs);
    # model = Unet;
    model = flutterSysGenNet;
    train_and_test(model, response, timeSys)


if __name__=='__main__':

    path = '/media/server/f8d02269-b24b-4f82-9d4d-6e3e4a6fe012/口..口/Duanshiqiang/' \
           'DLGenerateFRFHwHt/data/tur_T-10s_Fs-512';
    ResultSaveHomePath = '/media/server/f8d02269-b24b-4f82-9d4d-6e3e4a6fe012/口..口/Duanshiqiang/DLGenerateFRFHwHt/result';
    MAXDATASIZE = 2000;
    TRAIN_TEST_RATE = 0.2;
    BATCH_SIZE = int(MAXDATASIZE*(1-TRAIN_TEST_RATE));
    time,sampleRate = pathSplit(path)
    T = time;
    Fs = sampleRate;
    LayerNumber = 10
    NumberofFeatureChannel = 5
    timeLength = T*Fs;
    EPOCH = 50000;
    LR = 0.0001;

    main(path)




