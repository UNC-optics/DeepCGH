import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def gs(img, k):
    phi = np.random.rand(*list(img.shape)).astype(np.float32)
    while k:
        img_cf = img * np.exp(1.j * phi)
        slm_cf = np.fft.ifft2(img_cf)
        slm_cf = 1 * np.exp(1j * np.angle(slm_cf))
        img_cf = np.fft.fft2(slm_cf)
        phi = np.angle(img_cf)
        k -= 1
    return np.square(np.angle(img_cf))

def display_results(imgs, phases, recons, t):
    assert imgs.ndim == 4 and phases.ndim == 4 and recons.ndim == 4, "Dimensions don't match"
    for img, phase, recon in zip(imgs, phases, recons):
        if img.shape[-1] == 1:
            fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True, sharex=True)
            axs[0].imshow(np.squeeze(img), cmap='gray')
            axs[0].set_title('Target')
            axs[1].imshow(np.squeeze(phase), cmap='gray')
            axs[1].set_title('SLM Phase')
            axs[2].imshow(np.squeeze(recon), cmap='gray')
            axs[2].set_title('Simulation')
        else:
            fig, axs = plt.subplots(2, img.shape[-1] + 1, figsize = (3 * (img.shape[-1] + 1), 6), sharey = True, sharex = True)
            axs[0, -1].imshow(np.squeeze(phase))
            axs[0, -1].set_title('SLM Phase')
            for i in range(img.shape[-1]):
                axs[0, i].imshow(img[:, :, i], cmap='gray')
                axs[0, i].set_title('Target')
                axs[1, i].imshow(recon[:, :, i], cmap='gray')
                axs[0, i].set_title('Simulation')
        fig.suptitle('Inference time was {:.2f}ms'.format(t*1000), fontsize=16)

def get_propagate(data, model):
    shape = data['shape']
    zs = [-0.005*x for x in np.arange(1, (shape[-1]-1)//2+1)][::-1] + [0.005*x for x in np.arange(1, (shape[-1]-1)//2+1)]
    lambda_ = model['wavelength']
    ps = model['pixel_size']

    def __get_H(zs, shape, lambda_, ps):
        Hs = []
        for z in zs:
            x, y = np.meshgrid(np.linspace(-shape[1] // 2 + 1, shape[1] // 2, shape[1]),
                               np.linspace(-shape[0] // 2 + 1, shape[0] // 2, shape[0]))
            fx = x / ps / shape[0]
            fy = y / ps / shape[1]
            exp = np.exp(-1j * np.pi * lambda_ * z * (fx ** 2 + fy ** 2))
            Hs.append(exp.astype(np.complex64))
        return Hs

    def __prop__(cf_slm, H=None, center=False):
        if not center:
            H = tf.broadcast_to(tf.expand_dims(H, axis=0), tf.shape(cf_slm))
            cf_slm *= tf.signal.fftshift(H, axes=[1, 2])
        fft = tf.signal.ifftshift(tf.signal.fft2d(tf.signal.fftshift(cf_slm, axes=[1, 2])), axes=[1, 2])
        img = tf.cast(tf.expand_dims(tf.abs(tf.pow(fft, 2)), axis=-1), dtype=tf.dtypes.float32)
        return img

    def __phi_slm(phi_slm):
        i_phi_slm = tf.dtypes.complex(np.float32(0.), tf.squeeze(phi_slm, axis=-1))
        return tf.math.exp(i_phi_slm)

    Hs = __get_H(zs, shape, lambda_, ps)

    def propagate(phi_slm):
        frames = []
        cf_slm = __phi_slm(phi_slm)
        for H, z in zip(Hs, zs):
            frames.append(__prop__(cf_slm, tf.keras.backend.constant(H, dtype=tf.complex64)))

        frames.insert(shape[-1] // 2, __prop__(cf_slm, center=True))

        return tf.concat(values=frames, axis=-1)
    return propagate