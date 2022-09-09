#!/usr/bin/python

"""
vOICe - Seeing With Sound
https://www.seeingwithsound.com/

Original:  Derp Magurp - https://bitbucket.org/magurp244/audio-code/downloads/
Optimised: Jaime RS (Apr 2021)
"""

import math
import cv2
import numpy as np


class ImageToSound(object):

    TWO_PI = 2 * math.pi
    v = 340.0  # v = speed of sound (m/s)
    hs = 0.20  # hs = characteristic acoustical size of head (m)

    def __init__(self,
                 img_width=64,
                 freq_low=500,
                 freq_high=5000,
                 sampling_freq=11025,
                 sweep_time=1.05,
                 d='exponential',
                 delay=True):
        """

        :param img_width:
        :param freq_low:
        :param freq_high:
        :param sampling_freq: (int) Can be one of [8000, 11025, 16000, 22050, 32000, 44100, 48000, 88200, 96000, 192000]
        :param sweep_time:
        :param d:
        :param delay:
        """
        self.ns = np.floor(sampling_freq * sweep_time).astype(np.int)
        self.dt = 1.0 / sampling_freq
        self.sampling_freq = sampling_freq
        self.img_width = img_width
        self.m = self.ns / self.img_width
        self.scale = 0.5 / math.sqrt(self.img_width)
        self.img_height = self.img_width  # Assume square image
        self.play_obj = None

        # ---------------------------------------------------------
        # Set frequency distribution and random initial phase
        # ---------------------------------------------------------
        if d == 'exponential':
            self.w = self.TWO_PI * freq_low * np.power(1.0 * freq_high / freq_low, np.arange(0, self.img_height) / (self.img_height - 1))
        elif d == 'linear':
            self.w = self.TWO_PI * freq_low + self.TWO_PI * (freq_high - freq_low) * np.arange(0, self.img_height) / (self.img_height - 1)
        else:
            raise ValueError(f'Invalid frequency distribution {d}')

        self.rnd = (49297 % 233280) / 233280  # Why???? --JRS
        self.phi0 = self.TWO_PI * self.rnd
        self.tau1 = 0.5 / self.w[self.img_height - 1]
        self.tau2 = 0.25 * self.tau1 * self.tau1

        self.k = np.arange(0, self.ns)  # Indices of samples
        self.R = 1.0 * self.k / (self.ns - 1)  # Binaural attenuation/delay parameter
        self.theta = (self.R - 0.5) * self.TWO_PI / 3
        self.x = 0.5 * self.hs * (self.theta + np.sin(self.theta))
        self.tl = self.k * self.dt
        self.tr = self.k * self.dt
        if delay:
            self.tr += self.x / self.v  # Time delay model

        self.x = np.abs(self.x)
        self.sso = self.ssm = int(256 / 2)

    def process_image(self, image, stereo=True, diffr=True, fade=True, bspl=True):
        """

        :param image:
        :param stereo:
        :param diffr:
        :param fade:
        :param bspl: (bool) Time window — B-spline if True, rectangular if False
        :return:
        """

        hrtfl = np.ones((self.ns, self.img_height))
        hrtfr = np.ones((self.ns, self.img_height))

        if diffr:
            hrtf = self.TWO_PI * self.v / np.outer(self.x, self.w)
            hrtf[np.where(hrtf > 1.0)] = 1.0

            hrtfl[np.where(self.theta >= 0.0)] = hrtf[np.where(self.theta >= 0.0)]
            hrtfr[np.where(self.theta < 0.0)] = hrtf[np.where(self.theta < 0.0)]

        if fade:
            # Simple frequency-independent relative fade model
            hrtfl = (hrtfl.T * (1.0 - 0.7 * self.R)).T
            hrtfr = (hrtfr.T * (0.3 + 0.7 * self.R)).T

        image = cv2.flip(image, 0)  # The image needs to be flipped for the row indices to mean y-axis values

        # --------------------------------------------------------------------------
        # Generate audio (mono)
        # --------------------------------------------------------------------------
        if bspl:
            q = 1.0 * (self.k % self.m) / (self.m - 1)
            q2 = 0.5 * q * q

            a = []
            col_indices = (self.k / self.m).astype(np.uint8)
            for k_idx in self.k:  # FIXME: This weird convolution needs to be optimised with numpy operations

                j = col_indices[k_idx]

                if j == 0:
                    a.append((1.0 - q2[k_idx]) * image[:, j] +
                             q2[k_idx] * image[:, j + 1])

                elif j == self.img_width - 1:
                    a.append((q2[k_idx] - q[k_idx] + 0.5) * image[:, j - 1] +
                             (0.5 + q[k_idx] - q2[k_idx]) * image[:, j])

                else:
                    a.append((q2[k_idx] - q[k_idx] + 0.5) * image[:, j - 1] +
                             (0.5 + q[k_idx] - q[k_idx] ** 2) * image[:, j] +
                             q2[k_idx] * image[:, j + 1])

            a = np.array(a)

        else:
            repeats = [self.m] * image.shape[1]
            repeats[-1] += int((self.m % 1) * image.shape[1])  # When 'm' is not a whole number, we need to pad
            a = np.repeat(image, repeats, axis=1).T

            if a.shape != hrtfl.shape:
                # a_difference = hrtfl.shape[0] - a.shape[0]
                a = np.append(a, [a[-1]], axis=0)

                assert a.shape == hrtfl.shape, f'{a.shape}, {hrtfl.shape}'

        if not stereo:
            # TODO
            raise NotImplemented('Please use stereo for now')
            # audio = a.copy(order='C')
            # return audio

        # --------------------------------------------------------------------------
        # Left to right panning (stereo)
        # --------------------------------------------------------------------------

        # Left ---------------------------------------------------------------------
        sl = np.zeros(self.ns)
        sin_arg_l = np.outer(self.w, self.tl) + self.phi0
        sum_arg_l = hrtfl * a * np.sin(sin_arg_l).T
        sl += np.sum(sum_arg_l, axis=1)
        sl[np.where(self.k < self.m/5)] = (2.0 * self.rnd - 1.0) / self.scale  # Left "click"
        sl[np.where(self.tl < 0)] = 0.0

        ypl = zl = 0.0
        yl = self.tau1/self.dt + self.tau2/(self.dt*self.dt)
        yl = (sl + yl * ypl + self.tau2/self.dt * zl) / (1.0 + yl)
        yl *= 32767 / max(abs(yl))
        yl = yl.astype(np.int16)

        # Right ---------------------------------------------------------------------
        sr = np.zeros(self.ns)
        sin_arg_r = np.outer(self.w, self.tr) + self.phi0
        sum_arg_r = hrtfr * a * np.sin(sin_arg_r).T
        sr += np.sum(sum_arg_r, axis=1)
        sr[np.where(self.tr < 0)] = 0.0

        ypr = zr = 0.0
        yr = self.tau1/self.dt + self.tau2/(self.dt*self.dt)
        yr = (sr + yr * ypr + self.tau2/self.dt * zr) / (1.0 + yr)
        yr *= 32767 / max(abs(yr))
        yr = yr.astype(np.int16)

        # Put the two channels together --------------------------------------------
        audio = np.stack((yl, yr)).T.copy(order='C')

        return audio


if __name__ == '__main__':

    import simpleaudio
    import matplotlib.pyplot as plt

    # Dummy image
    image = np.zeros((64, 64))
    image[10:12, :30] = 100
    image[40:45, 50:] = 250

    img2sound = ImageToSound()
    audio = img2sound.process_image(image, bspl=False)

    # Play audio
    play_obj = simpleaudio.play_buffer(audio, 2, 2, img2sound.sampling_freq)

    # Show image
    fig = plt.figure()
    plt.imshow(image, cmap='gray')
    plt.show()

    # Wait to finish
    play_obj.wait_done()
