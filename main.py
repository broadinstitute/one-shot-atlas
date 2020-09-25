from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from data_generator import *
from models import *


NAME = 'coronal_depth_matching'
BATCH_SIZE = 16
NUM_EPOCHS = 2
PERIOD = 5
INIT_EPOCH = 0


def get_callbacks():
    tb_callback = TensorBoard(log_dir='./TensorBoard/' + NAME)
    filepath = 'checkpoints/weights-{epoch:02d}.hdf5'
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False, save_freq=PERIOD)
    return [tb_callback, checkpoint]


if __name__ == '__main__':
    import os    
    # get base folder for data (modify to point to your data)
    base_fld = './brain_images'
    assert os.path.isdir(base_fld)

    model = get_model()
    if INIT_EPOCH > 0:
        model.load_weights('checkpoints/weights-'+str(INIT_EPOCH)+'.hdf5')
        
    data_gen = DataGenerator(base_folder=base_fld, reshape_to=(model.input_shape[2], model.input_shape[3]),
                             batch_size=BATCH_SIZE)
    history = model.fit(data_gen, epochs=NUM_EPOCHS, workers=16, verbose=1,
                        callbacks=get_callbacks(), initial_epoch=INIT_EPOCH)
    model.save(NAME + '.h5')



