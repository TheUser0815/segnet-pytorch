import torch

def haar_wavelet(img):
    img_even = img[:,:,:,::2]
    img_odd = img[:,:,:,1::2]
    shallow_approximation = img_even + img_odd
    shallow_details = img_even - img_odd

    approx_even = shallow_approximation[:,:,::2,:]
    approx_odd = shallow_approximation[:,:,1::2,:]

    approximation = approx_even + approx_odd
    vertical = approx_even - approx_odd

    detail_even = shallow_details[:,:,::2,:]
    detal_odd = shallow_details[:,:,1::2,:]

    horizontal = detail_even + detal_odd
    diagonal = detail_even - detal_odd
    return torch.stack([approximation, horizontal, vertical, diagonal], 2)[:,0]

def extract_borders(img, pixelvariance=1):
    img = torch.round(img)
    left = img - torch.cat([img[:,:,:,:pixelvariance], img], 3)[:,:,:,:-1 * pixelvariance]
    left[left < 0.] = 0.
    right = img - torch.cat([img, img[:,:,:,-1 * pixelvariance:]], 3)[:,:,:,pixelvariance:]
    right[right < 0.] = 0.
    top = img - torch.cat([img[:,:,:pixelvariance], img], 2)[:,:,:-1 * pixelvariance]
    top[top < 0.] = 0.
    bottom = img - torch.cat([img, img[:,:,-1 * pixelvariance:]], 2)[:,:,pixelvariance:]
    bottom[bottom < 0.] = 0.
    shape = top + left + bottom + right
    shape[shape > 0.] = 1.
    return shape
