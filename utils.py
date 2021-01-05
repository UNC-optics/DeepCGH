import numpy as np
import matplotlib.pyplot as plt

def gs(img, k):
    phi = np.random.rand(img.shape)
    while K:
        img_cf = img * np.exp(1j * phi)
        slm_cf = np.fft.ifft2(img_cf)
        slm_cf = 1 * np.exp(1j * np.angle(slm_cf))
        img_cf = np.fft.fft2(slm_cf)
        phi = np.angle(img_cf)
        k -= 1
    return np.square(np.abs(img_cf))

def display_results(imgs, phases, recons):
    assert imgs.ndim == 4 and phases.ndim == 4 and recons.ndim == 4, "Dimensions don't match"
    for img, phase, recon in zip(imgs, phases, recons):
        if img.shape[-1] == 1:
            fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True, sharex=True)
            axs[0].imshow(np.squeeze(img), cmap='gray')
            axs[1].imshow(np.squeeze(phase), cmap='gray')
            axs[2].imshow(np.squeeze(recon), cmap='gray')
        else:
            fig, axs = plt.subplots(2, img.shape[-1] + 1, figsize = (3 * (img.shape[-1] + 1), 6), sharey = True, sharex = True)
            axs[-1].imshow(phase)
            for i in range(img.shape[-1]):
                axs[i].imshow(img[:, :, i], cmap='gray')
                axs[i + img.shape[-1]].imshow(recon[:, :, i], cmap='gray')
