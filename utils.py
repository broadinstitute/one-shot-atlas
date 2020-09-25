import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pickle
import numpy as np
import pandas as pd
from scipy import optimize
import umap
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


def format_num_to_str(num):
    """
        Cast number to three digit string.
        eg. 10 --> 010. 1 --> 001.
    """
    n_str = str(num)
    for _ in range(3 - len(n_str)):
        n_str = '0' + n_str
    return n_str


def get_image(ix, base_fld, atlas='p56c'):
    """
        Get image at posterior ix from atlas
        `p56c`, `ccfv3` or `ccfv2`.
    """
    str_ix = format_num_to_str(ix)
    path = os.path.join(base_fld, str_ix + '/' + atlas + '.jpg')
    img = cv2.imread(path)
    return img


def save_latent_space(data_frame, siamese_net, atlas='p56'):
    # generate and save the latent space of an atlas
    mask_1 = data_frame.df['plane'] == 'cor'
    mask_2 = data_frame.df.coordinate % 10 == 0
    mask_3 = data_frame.df.is_mask == False
    mask_4 = data_frame.df['atlas'] == atlas
    p56 = data_frame.df[mask_1 & mask_2 & mask_3 & mask_4].copy()
    p56 = p56.sort_values('coordinate')
    labels = []

    for index, vis in p56.iterrows():
        tmp = data_frame.get_image_u8_from_path(vis.path)
        tmp = np.reshape(tmp, [-1, siamese_net.input_shape[2], siamese_net.input_shape[3], 3])
        tmp = tmp.astype('float32') / 255.
        labels.append(vis.coordinate / 10)

        if vis.path == list(p56.path)[0]:
            test_img = tmp
        else:
            test_img = np.concatenate([test_img, tmp], axis=0)

    test_img = np.stack((test_img,)*2, axis=1)
    get_latent_space = K.function([siamese_net.layers[0].input],
                                  [siamese_net.get_layer("latent_space").get_output_at(1)])
    latent_space = get_latent_space([test_img])[0]

    with open(atlas+'.pkl', 'wb') as f:
        pickle.dump([latent_space, np.array(labels)], f)


def plot_predictions(siamese_net, data_frame, test_atlas='p56', reference_atlas='ccfv3', index=None, plot=True):
    with open(reference_atlas+'.pkl', 'rb') as f:
        latent_space, labels = pickle.load(f)

    labels = labels * 10

    mask_1 = data_frame.df['plane'] == 'cor'
    mask_2 = data_frame.df.is_mask == False
    mask_3 = data_frame.df['atlas'] == test_atlas
    test = data_frame.df[mask_1 & mask_2 & mask_3].copy()
    if index is None:
        test = test.sample(n=1)
    else:
        test = test.take([index])

    for pt in test.path:
        tmp = data_frame.get_image_u8_from_path(pt).astype('float32') / 255.
        tmp = np.reshape(tmp, [-1, siamese_net.input_shape[2], siamese_net.input_shape[3], 3])
        if pt == list(test.path)[0]:
            test_img = tmp
        else:
            test_img = np.concatenate([test_img, tmp], axis=0)

    ground_truth = []
    predictions = []
    n_elements = latent_space.shape[0] 
    
    get_latent_space = K.function([siamese_net.layers[0].input],
                                  [siamese_net.get_layer("latent_space").get_output_at(1)])
    new_input = Input([2, 1, 512])
    get_prediction = new_input
    for layer in siamese_net.layers[5:]:
        get_prediction = layer(get_prediction)
    get_prediction = Model(inputs=new_input, outputs=get_prediction)
    test_img = np.stack((test_img,)*2, axis=1)
    test_latent = get_latent_space(test_img)[0]
    
    for ix in range(n_elements):
        anchor = latent_space[ix]
        anchor = np.reshape(anchor, [1, 512])
        predictions.append(get_prediction(np.array([test_latent, anchor])))
        ground_truth.append(np.abs(labels[ix]-np.array(test.coordinate)) / 1320.)

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    def f(x, x0, x1, x2, x3, x4):
        y = np.zeros(x.shape)
        for i in range(len(y)):
            y[i] = fitfunc(x[i], x0, x1, x2, x3, x4)
        return y

    def fitfunc(x, x0, x1, x2, x3, x4):
        # y = np.abs(-x+x0)
        if x < x0:
            y = x2*x + x1-x2*x0
        else:
            y = x3*x + x1-x3*x0
        return y

    x = np.arange(n_elements) * 10 + labels[0]
    y = np.reshape(predictions, (n_elements,)) * 1310
    params, _ = optimize.curve_fit(f, x/1310, y/1310, [0.5, 0.1, -1, 1, -0.5])
    
    p_min = float(int(list(y).index(y.min())) * 10)
    
    delta_x1 = int(131*(- params[1]+params[2]*params[0])/params[2])*10 
    delta_x2 = int(131*(- params[1]+params[3]*params[0])/params[3])*10
    delta_fit = np.abs(delta_x1 - delta_x2)/2
    p_fit = float(int(params[0] * 131) * 10)
    
    if p_min < 40:
        p_min = 40
        
    if p_fit < 40:
        p_fit = 40
    
    if np.abs(p_min - p_fit) < 0:
        # print('Predicted depth:', p_fit, 'um')
        mask1 = data_frame.df['atlas'] == 'ccfv3'  # reference_atlas
        mask2 = data_frame.df['coordinate'] == p_fit
        mask3 = data_frame.df.is_mask == False
        mask4 = data_frame.df['plane'] == 'cor'
        pred_im = data_frame.df[mask1 & mask2 & mask3 & mask4].copy()
        case = 'fit'
        p = p_fit
    else:
        # print('Predicted depth:', p_min, 'um')
        mask1 = data_frame.df['atlas'] == 'ccfv3'  # reference_atlas
        mask2 = data_frame.df['coordinate'] == p_min + labels[0]
        mask3 = data_frame.df.is_mask == False
        mask4 = data_frame.df['plane'] == 'cor'
        pred_im = data_frame.df[mask1 & mask2 & mask3 & mask4].copy()
        case = 'min'
        p = p_min + labels[0]

    if plot:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
        axs[0].plot(x/10, y/10, 'x', label='model predictions')
        axs[0].plot(x/10, f(x/1310, *params)*131, '--', label='fit on predictions')
        axs[0].set_ylabel("PIR coordinate difference (x100 um)")
        axs[0].set_xlabel("PIR coordinate (x100 um)")
        axs[0].set_title(test_atlas + ' matching ' + reference_atlas)
        axs[0].legend()
        axs[1].imshow(test_img[0, 0, :, :, :])
        axs[1].axis('off')
        if not np.isnan(list(test.coordinate)[0]):
            axs[1].set_title('sample image ')
        else:
            axs[1].set_title('sample image' + ' (unregistered)')
        axs[2].imshow(data_frame.get_image_u8_from_path(list(pred_im.path)[0]).astype('float32') / 255.)
        axs[2].axis('off')
        # axs[2].set_title('best match (' + str(p) + ' +- ' + str(delta_fit) +' um)')
        axs[2].set_title('best match (' + str(p*10) + ' um)')
        plt.show()
        # fig.savefig('new_predictions/'+test_atlas+'/'+str(index)+'_'+reference_atlas+'.png')
        
    return p, x, ground_truth, params


def plot_umap(new_dataframe, siamese_net):
    marker_var = []
    size_var = []
    get_latent_space = K.function([siamese_net.layers[0].input],
                                  [siamese_net.get_layer("latent_space").get_output_at(1)])

    ix = 0
    mask_1 = new_dataframe.df['plane'] == 'cor'
    mask_2 = new_dataframe.df.is_mask == False
    mask_3 = new_dataframe.df['atlas'] == 'brainmaps'
    test = new_dataframe.df[mask_1 & mask_2 & mask_3].copy()
    test = test.take([ix])
    tmp = new_dataframe.get_image_u8_from_path(list(test.path)[0]).astype('float32') / 255.
    tmp = np.reshape(tmp, [-1, siamese_net.input_shape[2], siamese_net.input_shape[3], 3])
    tmp = np.stack((tmp,)*2, axis=1)
    latent_space = get_latent_space(tmp)[0]
    p1, _, _, _ = plot_predictions(siamese_net, new_dataframe, test_atlas='brainmaps',
                                   reference_atlas='ccfv3', index=ix, plot=False)
    p2, _, _, _ = plot_predictions(siamese_net, new_dataframe, test_atlas='brainmaps',
                                   reference_atlas='ccfv2', index=ix, plot=False)
    p3, _, _, _ = plot_predictions(siamese_net, new_dataframe, test_atlas='brainmaps',
                                   reference_atlas='p56', index=ix, plot=False)
    marker_var.append('test')
    size_var.append('big')
    labels = np.reshape(int(p3/10), (1,))

    with open('ccfv3.pkl', 'rb') as f:    
        tmp_latent, tmp_labels = pickle.load(f)        
    latent_space = np.concatenate([latent_space, tmp_latent])
    labels = np.concatenate([labels, tmp_labels])
    for i in tmp_labels:
        marker_var.append('ccfv3')
        if i == int(p1/10):
            size_var.append('big')
        else:
            size_var.append('small')

    with open('ccfv2.pkl', 'rb') as f:
        tmp_latent, tmp_labels = pickle.load(f)
    latent_space = np.concatenate([latent_space, tmp_latent])
    labels = np.concatenate([labels, tmp_labels])
    for i in tmp_labels:
        marker_var.append('ccfv2')
        if i == int(p2/10):
            size_var.append('big')
        else:
            size_var.append('small')

    with open('p56.pkl', 'rb') as f:
        tmp_latent, tmp_labels = pickle.load(f)
    latent_space = np.concatenate([latent_space, tmp_latent])
    labels = np.concatenate([labels, tmp_labels])
    for i in tmp_labels:
        marker_var.append('p56')
        if i == int(p3/10):
            size_var.append('big')
        else:
            size_var.append('small')

    embedding = umap.UMAP(n_neighbors=200).fit_transform(latent_space)

    plt.figure(figsize=(16, 10))
    mask = np.array(marker_var) == 'ccfv3'
    plt.scatter(
        x=embedding[mask, 0], y=embedding[mask, 1],
        c=labels[mask],
        marker='s',
        cmap="inferno",
        alpha=0.7
    )
    plt.colorbar().set_label('depth (x100 um)', weight='bold')
    mask = np.array(marker_var) == 'ccfv2'
    plt.scatter(
        x=embedding[mask, 0], y=embedding[mask, 1],
        c=labels[mask],
        marker='^',
        cmap="inferno",
        alpha=0.7
    )
    mask = np.array(marker_var) == 'p56'
    plt.scatter(
        x=embedding[mask, 0], y=embedding[mask, 1],
        c=labels[mask],
        marker='o',
        cmap="inferno",
        alpha=0.7
    )
    mask_1 = np.array(size_var) == 'big'
    mask_2 = np.array(marker_var) == 'test'
    plt.scatter(
        x=embedding[mask_1 & mask_2, 0], y=embedding[mask_1 & mask_2, 1],
        c='green',
        marker='X',
        s=200,
        alpha=1.
    )
    mask_1 = np.array(size_var) == 'big'
    mask_2 = np.array(marker_var) == 'ccfv3'
    plt.scatter(
        x=embedding[mask_1 & mask_2, 0], y=embedding[mask_1 & mask_2, 1],
        c='green',
        marker='s',
        s=200,
        alpha=1.
    )
    mask_1 = np.array(size_var) == 'big'
    mask_2 = np.array(marker_var) == 'ccfv2'
    plt.scatter(
        x=embedding[mask_1 & mask_2, 0], y=embedding[mask_1 & mask_2, 1],
        c='green',
        marker='^',
        s=200,
        alpha=1.
    )
    mask_1 = np.array(size_var) == 'big'
    mask_2 = np.array(marker_var) == 'p56'
    plt.scatter(
        x=embedding[mask_1 & mask_2, 0], y=embedding[mask_1 & mask_2, 1],
        c='green',
        marker='o',
        s=200,
        alpha=1.
    )

    markers = ['s', 'o', '^', 'X']
    lines = [Line2D([0], [0], color='black', linewidth=3, linestyle=' ', marker=m) for m in markers]
    labels = ['CCFv3', 'P56', 'CCFv2', 'Test Image']
    plt.legend(lines, labels, fontsize='large')
    plt.show()
