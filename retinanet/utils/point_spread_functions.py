"""!
Copyright (C) 2019 Bruno Gregorio - BIPG

    https://brunoggregorio.github.io \n
    https://www.bipgroup.dc.ufscar.br

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math
import cv2
import numpy as np
from skimage.draw import line
from scipy.stats import lognorm
from matplotlib import pyplot as plt
from sklearn.preprocessing import minmax_scale

"""!brief
@see
    - http://zeiss-campus.magnet.fsu.edu/print/basics/psf-print.html
    - https://hal.inria.fr/inria-00395534/document
    - https://arxiv.org/pdf/1809.01579.pdf
"""

def _normalize_kernel(kernel, mode='integral'):
    """!@brief
    Normalize the filter kernel.

    @param mode : {'integral', 'peak'}
        One of the following modes:
            * 'integral' (default)
                Kernel is normalized such that its integral = 1.
            * 'peak'
                Kernel is normalized such that its peak = 1.
    """

    if mode == 'integral':
        normalization = kernel.sum()
    elif mode == 'peak':
        normalization = kernel.max()
    else:
        raise ValueError("invalid mode, must be 'integral' or 'peak'")

    # Warn the user for kernels that sum to zero
    if normalization == 0:
        print('The kernel cannot be normalized because it sums to zero.')
    else:
        np.divide(kernel, normalization, kernel)

    return kernel


def _round_up_to_odd_integer(value):
    i = math.ceil(value)
    if i % 2 == 0:
        return i + 1
    else:
        return i


def _find_nearest_angle(center, angle):
    # Find the number of reasonable lines
    numDistinctLines = center * 4

    # Find the list of angles
    angle = math.fmod(angle, 180.0)
    validLineAngles = np.linspace(0, 180, numDistinctLines, endpoint=False)

    # Find the nearest valid angle
    idx = (np.abs(validLineAngles-angle)).argmin()
    angle = validLineAngles[idx]

    return angle


def AiryDiskKernel(radius):
    """!@brief
    2D Airy disk kernel.

    This kernel models the diffraction pattern of a circular aperture. This
    kernel is normalized to a peak value of 1.

    @param radius (float) : The radius of the Airy disk kernel (radius of the
                            first zero).

    @see
        LineKernel, CrossKernel, SnakeKernel

    References
    --------
    [1] : https://docs.astropy.org/en/stable/api/astropy.convolution.AiryDisk2DKernel.
    html#astropy.convolution.AiryDisk2DKernel
    [2] : https://svi.nl/AiryDisk
    [3] : https://en.wikipedia.org/wiki/Airy_disk
    [4] : https://www.lfd.uci.edu/~gohlke/code/psf.py.html
    [5] : https://www.sciencedirect.com/science/article/pii/B9780121197926501352
    (1) Electromagnetic diffraction in optical systems. II. Structure of the
        image field in an aplanatic system.
        B Richards and E Wolf. Proc R Soc Lond A, 253 (1274), 358-379, 1959.
    (2) Focal volume optics and experimental artifacts in confocal fluorescence
        correlation spectroscopy.
        S T Hess, W W Webb. Biophys J (83) 2300-17, 2002.
    (3) Electromagnetic description of image formation in confocal fluorescence
        microscopy.
        T D Viser, S H Wiersma. J Opt Soc Am A (11) 599-608, 1994.
    (4) Photon counting histogram: one-photon excitation.
        B Huang, T D Perroud, R N Zare. Chem Phys Chem (5), 1523-31, 2004.
        Supporting information: Calculation of the observation volume profile.
    (5) Gaussian approximations of fluorescence microscope point-spread function
        models.
        B Zhang, J Zerubia, J C Olivo-Marin. Appl. Optics (46) 1819-29, 2007.
    (6) The SVI-wiki on 3D microscopy, deconvolution, visualization and analysis.
        https://svi.nl/NyquistRate
    (7) Theory of Confocal Microscopy: Resolution and Contrast in Confocal
        Microscopy. http://www.olympusfluoview.com/theory/resolutionintro.html

    Example:
    --------
    Kernel response:

     .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from .utils import AiryDiskKernel
        airydisk_2D_kernel = AiryDiskKernel(10)
        plt.imshow(airydisk_2D_kernel, interpolation='none', origin='lower')
        plt.xlabel('x [pixels]')
        plt.ylabel('y [pixels]')
        plt.colorbar()
        plt.show()
    """
    # Define kernel size according to radius value
    k_size = _round_up_to_odd_integer(3*radius)

    # Set ranges where to evaluate the model
    if k_size % 2 == 0:  # even kernel
        xy_range = (-(int(k_size)) // 2 + 0.5, (int(k_size)) // 2 + 0.5)
    else:  # odd kernel
        xy_range = (-(int(k_size) - 1) // 2, (int(k_size) - 1) // 2 + 1)

    x = np.arange(*xy_range)
    y = np.arange(*xy_range)
    x, y = np.meshgrid(x, y)

    amplitude = 1
    x_0 = 0
    y_0 = 0

    try:
        from scipy.special import j1, jn_zeros
        _rz = jn_zeros(1, 1)[0] / np.pi # equals to 1.22
        _j1 = j1
    except ValueError:
        raise ImportError('AiryDisk2D model requires scipy > 0.11.')

    r = np.sqrt((x - x_0) ** 2 + (y - y_0) ** 2) / (radius / _rz)

    # Since r can be zero, we have to take care to treat that case
    # separately so as not to raise a numpy warning
    z = np.ones(r.shape)
    rt = np.pi * r[r > 0]
    z[r > 0] = (2.0 * _j1(rt) / rt) ** 2
    z *= amplitude

    # Normalize kernel values
    kernel = _normalize_kernel(z)

    #------------------------------------------------------------------------------
    # # Print kernel for debug purpose
    # from mpl_toolkits import mplot3d
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.plot_surface(x,y,z, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
    # plt.show()
    #------------------------------------------------------------------------------

    return kernel


def MotionKernel(k_size=3,
                 k_angle=0,
                 k_type=None):
    """!@brief
    Create a kernel simulating the motion in the image by using random params.

    @param k_size  : The kernel size.
    @param k_angle : The angle of motion in the kernel.
    @param k_type  : The kernel type {'line', 'curve'}.

    @return
        A kernel for motion simulation.
    """

    # Initialize kernel with zeros
    k = np.zeros((k_size,k_size))

    if k_type is None:
        types = ['line', 'curve']
        # Choose randomly the kernel type
        k_type = str(np.random.choice(types))

    if k_type == 'line':
        # Weighting the motion
        s=1
        x_w = np.linspace(lognorm.ppf(0.01, s), lognorm.ppf(0.99, s), k_size)
        rv = lognorm(s)

        # plt.plot(x_w, rv.logpdf(x_w), 'k-', lw=2, label='frozen pdf')
        # plt.show()

        line_1 = math.trunc(k_size/2)
        k[line_1:line_1+1,:] = rv.logpdf(x_w)

    elif k_type == 'curve':
        n_pts=100   # Number of points to create the function
        x = np.linspace(-0.1, 0.05, n_pts)
        y = np.cos(x)

        # Simulate the change of frequency in the kernel
        # delta_x = k_size/4
        delta_y = k_size/4

        # Discretizing function
        x = minmax_scale(x,feature_range=(0,k_size-1)).astype(int)
        y = minmax_scale(y,feature_range=(delta_y,k_size-delta_y)).astype(int)

        # plt.plot(x, y, 'k-', lw=2, label='frozen pdf')
        # plt.show()

        # Weighting the motion
        s=1
        x_w = np.linspace(lognorm.ppf(0.01, s), lognorm.ppf(0.99, s), n_pts)
        rv = lognorm(s)

        # plt.plot(x_w, rv.logpdf(x_w), 'k-', lw=2, label='frozen pdf')
        # plt.show()

        k[y,x] = rv.logpdf(x_w)

    else:
        raise ValueError("Invalid option for motion kernel.")

    # Blur kernel
    # k = cv2.GaussianBlur(k,(5,3),cv2.BORDER_DEFAULT)

    # Rotate kernel
    if k_angle:
        rows, cols = k.shape
        m = cv2.getRotationMatrix2D((math.trunc(cols/2), math.trunc(rows/2)), k_angle, 1)
        k = cv2.warpAffine(k, m, (cols, rows))

    # Normalize kernel
    kernel = k/k.sum()

    return kernel


def createPSFKernel(img_shape=None,
                    k_type=None,
                    k_size=None,
                    k_param=None):
    """!@brief
    Create a PSF kernel to simulate image motion.
    If all params are None the kernel is randomly created.

    The kernel types and usage are based on the paper [VillaPinto2015]_.
    .. [VillaPinto2015] Villa Pinto, C. H. and Gregorio da Silva, B. C.
       and Freire, P. G. L. and Bernardes, D. and Carvalho-Tavares, J.
       and Ferrari, R. J. "Deconvolucao cega aplicada a correcao de
       artefatos de movimento em imagens de video de microscopia intravital
       para deteccao automatica de leucocitos". Revista de Informatica
       Teorica e Aplicada: RITA, v. 22, p. 52-74, 2015.

    @param img_shape : The image shape to create kernel accordingly.
    @param k_type    : The type of the kernel [0:Gaussian, 1:AiryDisk, 2:Motion].
    @param k_size    : The size of the square kernel.
    @param k_param   : The kernel parameter (such as sigma or angle).

    @return
        The kernel as a matrix.
    """
    # Set kernel size
    if k_size is None or k_size == 0:
        if img_shape:
            min_side = min(img_shape[:2])
        else:
            min_side = 500  # Approximate 3~5 pixels kernel

        # Set kernel size as 0~1% of image min side
        random_state = np.random.RandomState(None)
        perc = random_state.uniform(0.0, 0.01)
        k_size = _round_up_to_odd_integer(min_side * perc)

        # Check kernel size
        if k_size < 3:
            return None

    elif k_size % 2 == 0:
        k_size += 1
    elif type(k_size) is not int:
        raise ValueError("Kernel size must be an integer positive number.")

    # Set kernel type
    k_types = [0,1,2]
    if k_type is None or k_type not in k_types:
        # Choose randomly the kernel type
        k_type = np.random.choice(k_types)

    ### Create the kernel according to the parameters

    # Gaussian kernel
    # -----------------
    if k_type==0:
        # k_param : sigma value
        if k_param is None:
            k_param = -1

        k_x = cv2.getGaussianKernel(k_size, k_param)
        k_y = cv2.getGaussianKernel(k_size, k_param)
        kernel = k_x * k_y.transpose()

    # Airy disk kernel
    # -----------------
    elif k_type==1:
        # k_param : radius (the first zero values)
        if k_param is None:
            k_param = k_size/3

        kernel = AiryDiskKernel(k_param)

    # Motion kernel
    # -----------------
    elif k_type==2:
        # k_param : rotation angle
        if k_param is None:
            random_state = np.random.RandomState(None)
            k_param = random_state.uniform(0, 360)

        kernel = MotionKernel(k_size=k_size, k_angle=k_param)

    else:
        raise ValueError("Kernel type must be in [0: Gaussian, 1: AiryDisk, 2: Motion('line', 'curve')].")

    #------------------------------------------------------------------------------
    # # Print kernel for debug purpose
    # types = {0:'Gaussian', 1:'Line', 2:'Cross', 3:'Snake'}
    # print("k_type: {}. k_size: {}. k_param: {}.".format(types[k_type], k_size, k_param))
    # nk = np.count_nonzero(kernel)
    # kk = kernel * nk
    #
    # plt.imshow(kernel, interpolation='none', origin='lower')
    # plt.xlabel('x [pixels]')
    # plt.ylabel('y [pixels]')
    # plt.colorbar()
    # plt.show()
    #------------------------------------------------------------------------------

    return kernel


def _create_3x3_line(angle):
    switch_angle = {
        0   : [1,0,1,2],
        45  : [2,0,0,2],
        90  : [0,1,2,1],
        135 : [0,0,2,2]
    }
    func = switch_angle.get(angle)
    return func


def _create_5x5_line(angle):
    switch_angle = {
        0     : [2,0,2,4],
        22.5  : [3,0,1,4],
        45    : [4,0,0,4],
        67.5  : [0,3,4,1],
        90    : [0,2,4,2],
        112.5 : [0,1,4,3],
        135   : [0,0,4,4],
        157.5 : [1,0,3,4]
    }
    func = switch_angle.get(angle)
    return func


def _create_7x7_line(angle):
    switch_angle = {
        0   : [3,0,3,6],
        15  : [4,0,2,6],
        30  : [5,0,1,6],
        45  : [6,0,0,6],
        60  : [6,1,0,5],
        75  : [6,2,0,4],
        90  : [0,3,6,3],
        105 : [0,2,6,4],
        120 : [0,1,6,5],
        135 : [0,0,6,6],
        150 : [1,0,5,6],
        165 : [2,0,4,6]
    }
    func = switch_angle.get(angle)
    return func


def _create_9x9_line(angle):
    switch_angle = {
        0      : [4,0,4,8],
        11.25  : [5,0,3,8],
        22.5   : [6,0,2,8],
        33.75  : [7,0,1,8],
        45     : [8,0,0,8],
        56.25  : [8,1,0,7],
        67.5   : [8,2,0,6],
        78.75  : [8,3,0,5],
        90     : [8,4,0,4],
        101.25 : [0,3,8,5],
        112.5  : [0,2,8,6],
        123.75 : [0,1,8,7],
        135    : [0,0,8,8],
        146.25 : [1,0,7,8],
        157.5  : [2,0,6,8],
        168.75 : [3,0,5,8]
    }
    func = switch_angle.get(angle)
    return func


def LineKernel(k_size, k_angle, linetype='full'):
    """!@brief
    Create a line kernel according to the params.

    @param k_size  : The size of the kernel.
    @param k_angle : The angle of the line in kernel.

    @return
        The kernel matrix.
    """

    # Assert the kernel size
    assert k_size in [3,5,7,9], 'Kernel size must be one of the values: {3,5,7,9}'

    # Find the kernel center
    k_center = int(math.floor(k_size/2))

    # Adjust to the proper angle value
    k_angle = _find_nearest_angle(k_center, k_angle)
    kernel = np.zeros((k_size, k_size), dtype=np.float32)

    # Create kernel points according to the params
    switch_size = {
        3 : _create_3x3_line(k_angle),
        5 : _create_5x5_line(k_angle),
        7 : _create_7x7_line(k_angle),
        9 : _create_9x9_line(k_angle)
    }
    linePoints = switch_size.get(k_size)

    # Adjust kernel type
    if linetype is 'right':
        linePoints[0] = k_center
        linePoints[1] = k_center
    if linetype is 'left':
        linePoints[2] = k_center
        linePoints[3] = k_center

    # Draw line patterns in the kernel
    rr,cc = line(linePoints[0], linePoints[1], linePoints[2], linePoints[3])
    kernel[rr,cc] = 1

    # Normalize kernel values
    kernel = _normalize_kernel(kernel)

    return kernel


def CrossKernel(k_size, k_angle):
    """!@brief
    Create a kernel with a cross as the acumulation of two line kernels.

    @param k_size  : The size of the kernel.
    @param k_angle : The angle of the lines in kernel.

    @return
        The kernel matrix.
    """
    assert k_angle <= 180, "The kernel angle must be less or equal to 180 degrees."

    k1 = LineKernel(k_size, k_angle)
    k2 = LineKernel(k_size, k_angle+90)
    kernel = k1+k2
    kernel[np.nonzero(kernel)] = 1

    # Normalize kernel values
    kernel = _normalize_kernel(kernel)

    return kernel


def SnakeKernel(k_size, k_pos):
    """!@brief
    Create a snake kernel according to the params.

    @param k_size : The size of the kernel.
    @param k_pos  : The position of the snake pattern.

    @return
        The kernel matrix.
    """

    # Assertion kernel size and position
    assert k_size in [5,7,9,11], 'Kernel size must be one of the values: {5,7,9,11}'
    assert k_pos   in [1,2,3,4],  'Kernel position must be one of the values: {1,2,3,4}'

    # Initialize kernel with zeros
    kernel = np.zeros((k_size,k_size))

    # Create kernel points according to the params
    switch_size = {
        5  : [[0,1,1,2,3,3,4], [3,2,4,2,0,2,1]],
        7  : [[0,0,0,1,1,2,3,4,5,5,6,6,6], [3,4,5,2,6,2,3,4,0,4,1,2,3]],
        9  : [[0,0,0,0,1,1,2,2,3,4,5,6,6,7,7,8,8,8,8], [4,5,6,7,3,8,3,8,3,4,5,0,5,0,5,1,2,3,4]],
        11 : [[0,0,0,1,1,2,2,3,4,5,6,7,8,8,9,9,10,10,10], [6,7,8,5,9,4,10,4,4,5,6,6,0,6,1,5,2,3,4]]
    }
    idx = switch_size.get(k_size)
    kernel[tuple(idx)] = 1

    # Normalize kernel values
    k1 = _normalize_kernel(kernel)

    # Create other kernels
    switch_pos = {
        1 : k1,
        2 : np.rot90(k1),  # rotation of 90 degrees
        3 : np.fliplr(k1), # flip horizontaly
        4 : np.fliplr(np.rot90(k1))
    }
    kernel = switch_pos.get(k_pos)

    return kernel
