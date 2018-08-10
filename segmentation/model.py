from keras import Input, Model
from keras.applications.resnet50 import ResNet50
from keras.layers import UpSampling2D, Conv2D
from keras.utils import plot_model


def seg_model(classes: int = 2) -> Model:
    img = Input((224, 224, 3))
    fe = ResNet50(include_top=False, input_tensor=img)
    for l in fe.layers:
        l.trainable = False

    x = fe(img)

    x = UpSampling2D()(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)

    x = UpSampling2D()(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)

    x = UpSampling2D()(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)

    x = UpSampling2D()(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)

    x = UpSampling2D()(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)

    x = Conv2D(classes, 1, activation='softmax')(x)

    model = Model(inputs=img, outputs=x)
    model.compile('adam', 'categorical_crossentropy')
    return model


if __name__ == '__main__':
    m = seg_model()
    m.summary()
    plot_model(m)
