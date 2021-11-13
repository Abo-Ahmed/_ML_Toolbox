

class res(basic_model): 

    def relu_bn(self,inputs: Tensor) -> Tensor:
        relu = ReLU()(inputs)
        bn = BatchNormalization()(relu)
        return bn

    def create_plain_net(self):
        
        inputs = Input(shape=(512, 512, 3))
        num_filters = 32
        
        t = BatchNormalization()(inputs)
        t = Conv2D(kernel_size=3,
                strides=1,
                filters=num_filters,
                padding="same")(t)
        t = self.relu_bn(t)
        
        num_blocks_list = [4, 4]
        for i in range(len(num_blocks_list)):
            num_blocks = num_blocks_list[i]
            for j in range(num_blocks):
                downsample = (j==0 and i!=0)
                t = Conv2D(kernel_size=3,
                        strides= (1 if not downsample else 2),
                        filters=num_filters,
                        padding="same")(t)
                t = self.relu_bn(t)
            num_filters *= 2
        
        t = AveragePooling2D(4)(t)
        t = Flatten()(t)
        outputs = Dense(10, activation='softmax')(t)
        
        model = Model(inputs, outputs)

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def build(self):
        super().build()
        self.model = self.create_plain_net()
        

    def program_2(self):
        self.load_model('/content/drive/MyDrive/CoLab/p_dan_pre_model/model-resnet_custom_v4.h5')
        self.summery_plot()
        configure.readTags("_master/networks/resnet_custom_v4/tags.txt")

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
            json_metadata_r = pickle.load(handle)
            return json_metadata_r

    def get_rating(id , json_metadata):
        for item in json_metadata :
            if item['id'] == id :
                return item['rating']
        return -1
    



