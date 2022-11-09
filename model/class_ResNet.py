

class ResNet(BasicModel): 

    def relu_bn(self,inputs: tf.Tensor) -> tf.Tensor:
        relu = keras_layers.ReLU()(inputs)
        bn = keras_layers.BatchNormalization()(relu)
        return bn

    def create_plain_net(self):
        
        inputs = tf.keras.Input(shape=(512, 512, 3))
        nFilters = 32
        
        t = keras_layers.BatchNormalization()(inputs)
        t = keras_layers.Conv2D(kernel_size=3,
                strides=1,
                filters=nFilters,
                padding="same")(t)
        t = self.relu_bn(t)
        
        nBlocksList = [4, 4]
        for i in range(len(nBlocksList)):
            nBlocks = nBlocksList[i]
            for j in range(nBlocks):
                downsample = (j==0 and i!=0)
                t = keras_layers.Conv2D(kernel_size=3,
                        strides= (1 if not downsample else 2),
                        filters=nFilters,
                        padding="same")(t)
                t = self.relu_bn(t)
            nFilters *= 2
        
        t = keras_layers.AveragePooling2D(4)(t)
        t = keras_layers.Flatten()(t)
        outputs = keras_layers.Dense(10, activation='softmax')(t)
        
        model = tf.keras.Model(inputs, outputs)

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def build(self):
        super().build()
        self.model = self.create_plain_net()
        

    def experiment_2(self):
        self.load_model('/content/drive/MyDrive/CoLab/p_dan_pre_model/model-resnet_custom_v4.h5')
        self.summery_plot()
        configure.readTags("_master_network/_networks/resnet_custom_v4/tags.txt")

    def special_predict(self):
        self.predict()
        e = handler.predict_y[0][-1]
        q = handler.predict_y[0][-2]
        s = handler.predict_y[0][-3]

        print("this Image: ")
        print(s)
        print(q)
        print(e)
        if(s > q and s > e):
            print("this image is save")
        elif (e > q and e > s):
            print("this image is pornographic")
        else:
            print("this image is quesionable")

    def load_metadata():
        with open('/content/drive/My Drive/dataSet/danbooru2019/metadata.pickle', 'rb') as handle:
            jsonMetadata = pickle.load(handle)
            return jsonMetadata

    def get_rating(id , jsonMetadata):
        for item in jsonMetadata :
            if item['id'] == id :
                return item['rating']
        return -1
    



