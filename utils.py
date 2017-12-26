import numpy as np


def concat_imgs(src_imgs, row_n, col_n):
    concated_img = np.empty((0, src_imgs.shape[1] * col_n, src_imgs.shape[3]))
    white_img = np.ones((src_imgs.shape[1], src_imgs.shape[2], src_imgs.shape[3]))
    for row_i in range(row_n):
        concated_row_img = np.empty((src_imgs.shape[1], 0, src_imgs.shape[3]))
        for col_i in range(col_n):
            count = row_i * col_n + col_i
            if count < len(src_imgs):
                concated_row_img = np.concatenate((concated_row_img, src_imgs[count]), axis=1)
            else:
                concated_row_img = np.concatenate((concated_row_img, white_img), axis=1)
        concated_img = np.concatenate((concated_img, concated_row_img), axis=0)
    return concated_img
