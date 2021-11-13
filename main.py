

# https://colab.research.google.com/drive/1weN1Kz-UcRYYFSnYW5ABB0RX77npMmWO?authuser=2#scrollTo=63C7VVdMx0wa

print(">> main module loaded ...")
execfile('/content/_master/handler/handler.py')

servent = handler()
# # 1 - configurations
handler.intial_configurations(True , True , '/content/_master') # mount , details , project path
print(servent)
# 2- loading and preparing dataset images
handler.read_data(None , 512 , 512) # datapath , width , height

# 3- model
# cnnSeq , cnnFunctional , res
# vgg16 , vgg16seq , lstm ,lstmBi , vggLstm , conv2Dlstm
# res - 1 - '/content/drive/MyDrive/CoLab/p_dan_pre_model/model-resnet_custom_v4.h5'
# handler.special_run("cnnFunctional","program_0")
handler.runModel("cnnSeq","program_0") # model , program
# handler.special_run("cnnFunctional","program_0") # model , program
# handler.runModel("res","program_0") # model , program
# handler.runModel("res","program_1") # model , program
# handler.runModel("lstm","program_0") # model , program
# handler.runModel("lstmBi","program_0") # model , program
# handler.runModel("vgg16","program_0") # model , program
# handler.runModel("vgg16Seq","program_0") # model , program
# handler.runModel("vggLstm","program_0") # model , program

# 4- show excution time
handler.final_configurations()
print(servent)