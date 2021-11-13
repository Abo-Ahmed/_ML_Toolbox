
# colab link
# https://colab.research.google.com/drive/1weN1Kz-UcRYYFSnYW5ABB0RX77npMmWO?authuser=2#scrollTo=63C7VVdMx0wa
print(">>> main module loaded ...")
execfile('/content/_master/handler/handler.py')
handler.intial_configurations(True , True , '/content/_master') # mount , details , project path

# loading and preparing dataset images
handler.read_data(None , 512 , 512) # datapath , width , height

# cnnSeq , cnnFunctional , res
# res - 1 - '/content/drive/MyDrive/CoLab/p_dan_pre_model/model-resnet_custom_v4.h5'
# vgg16 , vgg16seq , lstm ,lstmBi , vggLstm , conv2Dlstm
# handler.special_run("cnnFunctional","program_0")
handler.runModel("cnnSeq","program_0") # model , program
handler.runModel("cnnFunctional","program_0") # model , program
handler.runModel("res","program_0") # model , program
handler.runModel("res","program_2") # model , program
handler.runModel("lstm","program_0") # model , program
handler.runModel("lstmBi","program_0") # model , program
handler.runModel("vgg16","program_0") # model , program
handler.runModel("vgg16Seq","program_0") # model , program
handler.runModel("vggLstm","program_0") # model , program

handler.final_configurations()
