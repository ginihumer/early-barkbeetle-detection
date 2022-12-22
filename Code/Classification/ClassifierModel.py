import keras
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, GlobalMaxPooling2D

class MyClassifier():
    def __init__(self, base_model):
        
        self.base_model = base_model
    
    def get_transferred_model(self, weight_path=None, classes=2):
    
        # transfer learning
        for layer in self.base_model.layers:
            layer.trainable = False

        x = self.base_model.output
        # x = layers.Conv2D(kernel_size=3, filters=8, strides=1)(x)
        x = GlobalMaxPooling2D(name='max_pool')(x) #x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dropout(0.4)(x)

        predictions = Dense(classes, activation='softmax')(x)
        model = Model(inputs=self.base_model.input, outputs=predictions)

        #optimizer = keras.optimizers.Adadelta()
        optimizer = keras.optimizers.Adam()
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=optimizer, #keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        if weight_path:
            model.load_weights(weight_path)
        return model
    