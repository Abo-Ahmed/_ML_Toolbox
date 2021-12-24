
# colab link
# https://colab.research.google.com/drive/1weN1Kz-UcRYYFSnYW5ABB0RX77npMmWO?authuser=2#scrollTo=63C7VVdMx0wa
# https://colab.research.google.com/drive/1QA3ufl7lur76WzBhuwerLdyjLj-8hoHC?authuser=2

from dataset.dataset import dataset


print(">>> main module loaded ...")
execfile('/content/_master/handler/handler.py')
handler.intial_configurations(True , True , '/content/_master') # mount , details , project path


x = dataset.read_folder_images('/content/drive/MyDrive/eng-mahmoud/dataSet/danbooru2019/images/512px/0000',0)
print(x)
print(x.shape)

y = dataset.get_prediction_matrix('/content/drive/MyDrive/eng-mahmoud/dataSet/danbooru2019/images/512px/0000',0)
print(y)
print(y.shape)

# loading and preparing dataset images
# handler.read_data(None , 224 , 224) # datapath , width , height

# handler.run_models(["CnnSeq"        , "CnnFunctional"   , "ResNet" , 
#                     "Vgg16"         , "vgg16Seq"        , "VggLstm",
#                     "LstmConv2d"    , "Lstm"            , "LstmBi" , "VggBiLstm"])

# handler.special_run("Vgg16","program_1")
# handler.special_run("VggBiLstm","program_1")

handler.final_configurations()
