
# colab link
# https://colab.research.google.com/drive/1weN1Kz-UcRYYFSnYW5ABB0RX77npMmWO?authuser=2#scrollTo=63C7VVdMx0wa
# https://colab.research.google.com/drive/1QA3ufl7lur76WzBhuwerLdyjLj-8hoHC?authuser=2

print(">>> main module loaded ...")
execfile('/content/_master/handler/handler.py')
handler.intial_configurations(True , True , '/content/_master') # mount , details , project path

# loading and preparing dataset images
handler.read_data(None , 224 , 224) # datapath , width , height

# handler.run_models(["CnnSeq"        , "CnnFunctional"   , "ResNet" , 
#                     "Vgg16"         , "vgg16Seq"        , "VggLstm",
#                     "LstmConv2d"    , "Lstm"            , "LstmBi" , "VggBiLstm"])

handler.special_run("LstmConv2d","program_0")

handler.final_configurations()
