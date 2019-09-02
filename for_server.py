import cv2
from keras.models import load_model,Model
from keras.applications import ResNet50
from keras.layers import Dense
INPUT_WIDTH=224
def get_resnet():
    res_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(INPUT_WIDTH, INPUT_WIDTH, 3),
        pooling='avg'
    )

    prediction = Dense(units=101,
                       kernel_initializer='he_normal',
                       use_bias=False,
                       activation='softmax',
                       name='pred_age')(res_model.output)

    res_model = Model(inputs=res_model.input, outputs=prediction)
    return res_model
def get_full_model():
    res_model = get_resnet()

    last_res_layer = res_model.get_layer(index=-2)

    base_model = Model(inputs=res_model.input,outputs=last_res_layer.output)
    prediction = Dense(1, kernel_initializer='normal')(base_model.output)

    final_model = Model(inputs = base_model.input,outputs = prediction)
    return final_model



