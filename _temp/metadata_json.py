import numpy as np
import json

jsons = [          
          '/201700.json' , 
          '/201701.json' , 
          '/201702.json' , 
          '/201703.json' , 
          '/201704.json' , 
          '/201705.json' , 
          '/201706.json' , 
          '/201707.json' , 
          '/201708.json' , 
          '/201709.json' , 
          '/201710.json' ,
          '/201711.json' , 
          '/201712.json' , 
          '/201713.json' , 
          '/201714.json' , 
          '/201715.json' , 
          '/201716.json'  ]

global_path = '/content/content/danbooru-metadata/TAI-metadata'
json_objects = []
error_counter = 0 


for k in range (len(jsons)):  
  
    texts = open(global_path + jsons[k], 'r').read() .split('\n')   
    print('check {}: {} with items: {}'.format(k,global_path +  jsons[k] , len(texts) ) ) 
    
    for i in range(len(texts)-1):
      try:
        temp = json.loads(texts[i])
       
        if  ( int(temp['id'][-3:]) ) < 151  :
          
          json_objects.append(temp)
          
          with open("test.txt", "a") as myfile:
                myfile.write(texts[i] + '\n')
              
      except:
        error_counter = error_counter + 1
  
    
print('counter: {}'.format(len(json_objects)))
print('error_counter: {}'.format(error_counter))
print(json_objects[1000])
