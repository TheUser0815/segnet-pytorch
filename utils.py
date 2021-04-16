import torch

def haar_wavelet(img):
    img_even = img[:,:,:,::2]
    img_odd = img[:,:,:,1::2]
    shallow_approximation = img_even + img_odd
    shallow_details = img_even - img_odd

    approx_even = shallow_approximation[:,:,::2,:]
    approx_odd = shallow_approximation[:,:,1::2,:]

    vertical = approx_even + approx_odd
    diagonal = approx_even - approx_odd

    detail_even = shallow_details[:,:,::2,:]
    detal_odd = shallow_details[:,:,1::2,:]

    approximation = detail_even + detal_odd
    horizontal = detail_even - detal_odd
    return torch.stack([approximation, horizontal, vertical, diagonal], 2)[:,0]