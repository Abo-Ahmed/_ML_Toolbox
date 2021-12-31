
# colab link
# https://colab.research.google.com/drive/1QA3ufl7lur76WzBhuwerLdyjLj-8hoHC?authuser=2


print(">>> main module loaded ...")
execfile('/content/_master/handler/handler.py')
handler.intial_config(True , True , '/content/_master') # mount , details , project path
handler.dataset_config( handler.sPath , 224 , 224 , 30 , 1 , True) # path ,width , height , batchSize , startBatch , isColored

folder.download_image(33 ,100)

# # print(rating.values[1315650])
# handler.read_real('/content/drive/MyDrive/eng-mahmoud/dataSet/danbooru2019/images/512px/0000')
# # handler.read_sample( None)  

# # handler.run_models(["CnnSeq"      , "CnnFunctional" , "ResNet" , "Vgg16"  , "vgg16Seq" ,
# #                     "VggLstm"     , "LstmConv2d"    , "Lstm"   , "LstmBi" , "VggBiLstm" ] )
# handler.special_run("VggBiLstm","program_1")
# # handler.special_run("CnnFunctional","program_0")

handler.final_config()
