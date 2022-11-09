
# colab link
# https://colab.research.google.com/drive/1QA3ufl7lur76WzBhuwerLdyjLj-8hoHC?authuser=2
# https://colab.research.google.com/notebooks/pro.ipynb

print(">>> main module loaded ...")
execfile('/content/_master_network/handler/handler.py')

# mount , details , project path
handler.intial_config(True, True, '/content/_master_network')

# path ,width , height , batchSize , startBatch , isColored
handler.dataset_config(handler.ePath, 70, 70, 30, 1, False)

handler.read_real()
# handler.read_sample(None)

# handler.run_models(["CnnSeq"      , "CnnFunctional" , "ResNet" , "Vgg16"  , "vgg16Seq" ,
#                     "VggLstm"     , "LstmConv2d"    , "Lstm"   , 
#                     "LstmBi" , "VggBiLstm"  , "Transformer"] )

# handler.special_run("VggBiLstm","program_1")
handler.special_run("Transformer","program_0")

handler.final_config()
