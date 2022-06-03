import numpy as np


def color_masking(img, rgb):
    r, g, b = rgb
    return np.logical_and(np.logical_and(img[:, :, 0] == r, img[:, :, 1] == g), img[:, :, 2] == b)


def logical_or_masks(mask_list):
    mask_all = np.zeros_like(mask_list[0], dtype=bool)
    for mask in mask_list:
        mask_all = np.logical_or(mask_all, mask)
    return mask_all


def seg2mask(seg):
    img_numpy = np.array(seg)

    m_background = color_masking(img_numpy, (0, 0, 0))
    m_skin = color_masking(img_numpy, (204, 0, 0))
    m_l_brow = color_masking(img_numpy, (76, 153, 0))
    m_r_brow = color_masking(img_numpy, (204, 204, 0))
    m_l_eye = color_masking(img_numpy, (51, 51, 255))
    m_r_eye = color_masking(img_numpy, (204, 0, 204))
    m_eye_g = color_masking(img_numpy, (0, 255, 255))
    m_l_ear = color_masking(img_numpy, (255, 204, 204))
    m_r_ear = color_masking(img_numpy, (102, 51, 0))
    m_ear_r = color_masking(img_numpy, (255, 0, 0))
    m_nose = color_masking(img_numpy, (102, 204, 0))
    m_mouth = color_masking(img_numpy, (255, 255, 0))
    m_u_lip = color_masking(img_numpy, (0, 0, 153))
    m_l_lip = color_masking(img_numpy, (0, 0, 204))
    m_neck = color_masking(img_numpy, (255, 51, 153))
    m_neck_l = color_masking(img_numpy, (0, 204, 204))
    m_cloth = color_masking(img_numpy, (0, 51, 0))
    m_hair = color_masking(img_numpy, (255, 153, 51))
    m_hat = color_masking(img_numpy, (0, 204, 0))

    # gen mask for using in the model
    mask_face = logical_or_masks([m_skin, m_l_ear, m_r_ear])
    mask_hair = logical_or_masks([m_hair, m_hat])
    mask_eye = logical_or_masks([m_l_brow, m_r_brow, m_l_eye, m_r_eye, m_eye_g])
    mask_nose = logical_or_masks([m_nose])
    mask_lip = logical_or_masks([m_u_lip, m_l_lip])
    mask_tooth = logical_or_masks([m_mouth])
    mask_head = logical_or_masks([m_skin, m_l_ear, m_r_ear, m_l_brow, m_r_brow, m_l_eye, 
                                    m_r_eye, m_eye_g, m_nose, m_u_lip, m_l_lip, m_mouth, m_neck, m_neck_l])
    mask_background = logical_or_masks([m_background, m_cloth, m_hat, m_hair])
    
    # merge masks 
    masks = np.array([mask_face, mask_eye, mask_nose, mask_lip, mask_tooth])
    mask_head = np.array([mask_head])
    mask_background = np.array([mask_background])
    return masks, mask_head, mask_background
