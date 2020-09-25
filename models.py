import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, AveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import densenet
from tensorflow.keras.optimizers import Adam


def get_trainable_encoder():
    """build convnet to fine tune in each siamese 'leg'"""
    train_net = Sequential(name='latent_space')
    train_net.add(AveragePooling2D(pool_size=(3, 3), padding='valid'))
    train_net.add(Flatten())
    train_net.add(Dense(1024, activation='relu'))
    train_net.add(Dense(512, activation='sigmoid'))
    return train_net


def get_model():
    input_shape = (2, 224, 224, 3)
    input_pair = Input(input_shape, name='image_pair_input')

    lft_layer = Lambda(lambda x: x[:, 0, ...], name='siamese_branch_1')
    rgt_layer = Lambda(lambda x: x[:, 1, ...], name='siamese_branch_2')
    lft_input = lft_layer(input_pair)
    rgt_input = rgt_layer(input_pair)
    
    # get pretrained model from Keras applications and freeze weights
    encoder = densenet.DenseNet121(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    for layer in encoder.layers:
        layer.trainable = False
    layer.trainable = True  # unfreeze last layer

    fine_tune = get_trainable_encoder()

    # build the two "siamese" networks
    lft_encoded = fine_tune(encoder(lft_input))
    rgt_encoded = fine_tune(encoder(rgt_input))

    l1_layer = Lambda(lambda xs: K.abs(xs[0] - xs[1]), name='discriminator_input')
    l1_distance = l1_layer([lft_encoded, rgt_encoded])

    output = Dense(1, activation='hard_sigmoid', name='depth_prediction')(l1_distance)
    model = Model(inputs=input_pair, outputs=output, name='siamese_network')

    model.compile(loss='mean_squared_error', optimizer=Adam())
    return model
