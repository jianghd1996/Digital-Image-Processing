import cv2
import numpy as np
import math as mt
import matplotlib.pyplot as plt

def RGB2HSI(rgb_img):
    '''
        RGB image 2 HSI image
    '''
    rgb_img = np.array(rgb_img, dtype="float32")
    n, m = rgb_img.shape[0], rgb_img.shape[1]
    hsi_img = rgb_img.copy()
    B, G, R = cv2.split(rgb_img)
    [B, G, R] = [i / 255.0 for i in ([B, G, R])]
    H = np.zeros((n, m))
    S = np.zeros((n, m), dtype="float32")
    I = (R + G + B) / 3.0
    for i in range(n):
        x = np.sqrt((R[i] - G[i]) ** 2 + (R[i] - B[i]) * (G[i] - B[i]))
        theta = np.arccos(0.5 * (R[i] - B[i] + R[i] - G[i]) / x)
        h = np.zeros(m)
        h[B[i] <= G[i]] = theta[B[i] <= G[i]]
        h[G[i] < B[i]] = 2 * mt.pi-theta[G[i] < B[i]]
        h[x == 0] = 0
        H[i] = h / (2 * mt.pi)
    for i in range(n):
        Min = []
        for j in range(m):
            arr = [B[i][j], G[i][j], R[i][j]]
            Min.append(np.min(arr))
        Min = np.array(Min)
        S[i] = 1 - Min * 3 / (R[i]+B[i]+G[i])
        S[i][R[i] + B[i] + G[i] == 0] = 0
    hsi_img[:, :, 0] = H*255
    hsi_img[:, :, 1] = S*255
    hsi_img[:, :, 2] = I*255
    return hsi_img

def HSI2RGB(hsi_img):
    '''
        HSI image 2 RGB image
    '''
    n, m = hsi_img.shape[0], hsi_img.shape[1]
    rgb_img = hsi_img.copy()
    H, S, I = cv2.split(hsi_img)
    [H, S, I] = [i / 255.0 for i in ([H,S,I])]
    R, G, B = H, S, I
    for i in range(n):
        h = H[i] * 2 * mt.pi
        a1 = h >= 0
        a2 = h < 2 * mt.pi / 3
        a = a1 & a2
        tmp = np.cos(mt.pi / 3 - h)
        b = I[i] * (1 - S[i])
        r = I[i] * (1 + S[i] * np.cos(h) / tmp)
        g = 3 * I[i] - r - b
        B[i][a] = b[a]
        R[i][a] = r[a]
        G[i][a] = g[a]
        a1 = h >= 2 * mt.pi / 3
        a2 = h < 4 * mt.pi / 3
        a = a1 & a2
        tmp = np.cos(mt.pi - h)
        r = I[i] * (1 - S[i])
        g = I[i] * (1 + S[i] * np.cos(h - 2 * mt.pi / 3) / tmp)
        b = 3 * I[i] - r - g
        R[i][a] = r[a]
        G[i][a] = g[a]
        B[i][a] = b[a]
        a1 = h >= 4 * mt.pi / 3
        a2 = h < 2 * mt.pi
        a = a1 & a2
        tmp = np.cos(5 * mt.pi / 3 - h)
        g = I[i] * (1 - S[i])
        b = I[i] * (1 + S[i] * np.cos(h - 4 * mt.pi / 3) / tmp)
        r = 3 * I[i] - g - b
        B[i][a] = b[a]
        G[i][a] = g[a]
        R[i][a] = r[a]
    rgb_img[:, :,0] = B*255
    rgb_img[:, :,1] = G*255
    rgb_img[:, :,2] = R*255
    return rgb_img

def visual(x, name=""):
    '''
        Visual distribution of color hist
    '''
    print("Min : {} Max : {}".format(x.min(), x.max()))
    y = x.reshape(-1)
    plt.hist(y, bins=255)
    plt.xlim(0, 260)
    plt.savefig("{}_hist.png".format(name))
    plt.close()

def calcu_mapping(x):
    '''
        Calculate equalize histogram mapping with color distribution x.
    '''
    tot = sum(x)
    pix = []
    for i in range(256):
        if x[i] != 0:
            pix.append([i, x[i]])
    mapping = dict()

    if len(pix) == 1:
        mapping[pix[0][0]] = pix[0][0]
        return mapping

    # Optimize too much black or white
    while True:
        flag = 0
        for i in range(len(pix)):
            if pix[i][1] / tot > 0.1:
                pix[i][1] = tot * 0.1
                x[pix[i][0]] = pix[i][1]
                flag = 1
            elif pix[i][1] / tot > 0.01 and (pix[i][0] == 0 or pix[i][0] == 255):
                pix[i][1] = tot * 0.01
                x[pix[i][0]] = pix[i][1]
                flag = 1
        tot = sum(x)
        if flag == 0:
            break

    current = 0
    N = len(pix)
    for i in range(N):
        ratio = min(i, 1)
        l = int(255.0 * current / tot)
        current += pix[i][1]
        r = int(255.0 * current / tot)
        p = int(l + (r-l) * ratio)
        mapping[pix[i][0]] = p

    return mapping

def myequalizeHist(img):
    '''
        Histogram Equalize
    '''
    img = np.uint8(img)
    e_img = img.copy()
    pix = []
    count = [0] * 256
    for i in range(len(e_img)):
        for j in range(len(e_img[0])):
            pix.append([img[i][j], i, j])
            count[img[i][j]] += 1
    mapping = calcu_mapping(count)
    pix.sort()
    for i in range(len(pix)):
        e_img[pix[i][1]][pix[i][2]] = mapping[pix[i][0]]
    return e_img

def MyEqualize(img, mode):
    '''
        Deal with GRAY and RGB, HSI separately
    '''
    if mode == "GRAY":
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = myequalizeHist(img)
        visual(img, "raw")
        visual(result, "my")
    elif mode == "RGB":
        (b, g, r) = cv2.split(img)
        bH = myequalizeHist(b)
        gH = myequalizeHist(g)
        rH = myequalizeHist(r)
        visual(r, "raw")
        visual(rH, "my")
        result = cv2.merge((bH, gH, rH))
    elif mode == "HSI":
        hsi_img = RGB2HSI(img)
        (h, s, i) = hsi_img[:, :, 0], hsi_img[:, :, 1], hsi_img[:, :, 2]
        hH = h
        sH = s
        iH = myequalizeHist(i)
        visual(i, "raw")
        visual(iH, "my")
        hsi_img = np.stack((hH, sH, iH), axis=2)
        result = HSI2RGB(hsi_img)

    return result

def laplacian_filter(img, kernel):
    '''
        Apply laplacian filter to image with kernel.
    '''
    laplacian = np.array(img, dtype="int")
    n, m = laplacian.shape
    for i in range(n):
        for j in range(m):
            v = 0
            for l in range(3):
                for r in range(3):
                    v += kernel[l][r] * int(img[max(0, min(n - 1, i + l - 1))][max(0, min(m - 1, j + r - 1))])
            laplacian[i][j] = v
    return laplacian

def MyLaplacian(img, mode):
    '''
        Deal with GRAY and RGB separately
    '''
    if "1" in mode:
        kernel = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    else:
        kernel = [[1, 1, 1], [1, -8, 1], [1, 1, 1]]

    if len(img.shape) == 2:
        laplacian = laplacian_filter(img, kernel)
        norm = np.uint8(255 * (laplacian - laplacian.min()) / (laplacian.max() - laplacian.min()))
    else:
        (b, g, r) = cv2.split(img)
        b_l, g_l, r_l = laplacian_filter(b, kernel), laplacian_filter(g, kernel), laplacian_filter(r, kernel)
        laplacian = cv2.merge((b_l, g_l, r_l))
        norm = laplacian.copy()
        for i in range(3):
            sub = norm[:, :, i]
            norm[:, :, i] = np.uint8(255 * (sub - sub.min()) / (sub.max() - sub.min()))

    return np.clip(np.array(img, dtype="int")-laplacian, 0, 255), laplacian, norm

def FFT(img):
    fft = np.fft.fft2(img)

    ifft = np.fft.ifft2(fft)
    return 20 * np.log(np.abs(np.fft.fftshift(fft))), np.abs(ifft)

def Matching(img):
    width, height = img.shape[0], img.shape[1]
    w = width // 5
    h = height // 5

    p = (np.random.randint(width-w), np.random.randint(height-h))
    template = img[p[0]:p[0]+w, p[1]:p[1]+h]
    template_img = img.copy()
    cv2.rectangle(template_img, p, (p[0]+w, p[1]+h), 255, 2)

    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

    single_img = img.copy()

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(single_img, top_left, bottom_right, 255, 2)

    multi_img = img.copy()

    threshold = 0.8
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        bottom_right = (pt[0] + w, pt[1] + h)
        cv2.rectangle(multi_img, pt, bottom_right, 255, 2)

    return template_img, single_img, multi_img

def IdealHighPassFiltering(f_shift, D0):
    m = f_shift.shape[0]
    n = f_shift.shape[1]
    h1 = np.zeros((m, n))
    x0 = np.floor(m/2)
    y0 = np.floor(n/2)
    for i in range(m):
        for j in range(n):
            D = np.sqrt((i - x0)**2 + (j - y0)**2)
            if D >= D0:
                h1[i][j] = 1
    result = np.multiply(f_shift, h1)
    return result

def GaussLowPassFiltering(f_shift, D0):
    m = f_shift.shape[0]
    n = f_shift.shape[1]
    h1 = np.zeros((m, n))
    x0 = np.floor(m/2)
    y0 = np.floor(n/2)
    for i in range(m):
        for j in range(n):
            D = np.sqrt((i - x0)**2 + (j - y0)**2)
            h1[i][j] = np.exp((-1)*D**2/2/(D0**2))
    result = np.multiply(f_shift, h1)
    return result

def GFLP(img, r):
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)
    IHPF = np.fft.ifftshift(IdealHighPassFiltering(f_shift, r))
    GLPF = np.fft.ifftshift(GaussLowPassFiltering(f_shift, r))
    return np.abs(np.fft.ifft2(IHPF)), np.abs(np.fft.ifft2(GLPF))

def DCT(img):
    if len(img.shape) == 2:
        img = np.float32(img)
        dct = cv2.dct(img)

        idct = cv2.idct(dct)
    else:
        img = np.float32(img)
        dct = img.copy()
        idct = img.copy()

        for i in range(3):
            dct[:, :, i] = cv2.dct(img[:, :, i])
            idct[:, :, i] = cv2.idct(dct[:, :, i])

    return dct, idct

def Edge(img):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    dst = cv2.Canny(img, 50, 200, None, 3)

    lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

    hough = dst.copy()

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv.line(dst, pt1, pt2, 255, 3, cv.LINE_AA)

    return sobel_x, sobel_y, hough

def morphology(img, mode):
    kernel = np.ones((5, 5), np.uint8)
    if mode == "erode":
        return cv2.erode(img, kernel)
    elif mode == "dilation":
        return cv2.dilate(img, kernel)
    elif mode == "opening":
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    elif mode == "closing":
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    else:
        return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

def add_noise(img, mode):
    '''
    Add gaussian noise to image
    '''
    if mode == "gaussian":
        noise_size = [100, 400, 800, 1600]
        var = np.random.randint(len(noise_size))
        var = noise_size[var]
        print("noise var {}".format(var))
        if len(img.shape) == 2:
            noise = np.random.normal(0, var ** 0.5, img.shape)
            img = np.clip(img + noise, 0, 255)
        else:
            for i in range(3):
                noise = np.random.normal(0, var ** 0.5, img.shape[:2])
                img[:, :, i] = np.clip(img[:, :, i] + noise, 0, 255)

        return np.uint8(img), var

def myarithemeticblur(img, kernel_size):
    '''
        Arithemetic image blurring
    '''
    K = int(kernel_size // 2)

    blur_img = np.zeros(img.shape)

    n, m = img.shape
    for i in range(n):
        for j in range(m):
            blur_img[i][j] = np.mean(img[max(i-K, 0): min(i+K+1, n), max(j-K, 0) : min(j+K+1, m)])

    return np.uint8(blur_img)

def geomean(xs):
    '''
        Geometric mean operator
    '''
    xs = np.clip(xs, 0.1, 255)

    return mt.exp(mt.fsum(mt.log(x) for x in xs) / len(xs))

def mygeometricblur(img, kernel_size):
    '''
        Geometric image blurring
    '''

    K = int(kernel_size // 2)

    blur_img = np.zeros(img.shape, dtype="float32")

    n, m = img.shape
    for i in range(n):
        for j in range(m):
            blur_img[i][j] = geomean(img[max(i - K, 0): min(i + K + 1, n), max(j - K, 0): min(j + K + 1, m)].reshape(-1))

    return np.uint8(blur_img)

def myadaptiveblur(img, kernel_size, noise_var=1000):
    '''
        Adaptive image blurring
    '''

    if noise_var == 0:
        return img

    K = int(kernel_size // 2)

    blur_img = np.array(img, dtype="float32")

    n, m = img.shape

    for i in range(n):
        for j in range(m):
            if i < K or i-K+kernel_size > n-1 or j < K or j-K+kernel_size > m-1:
                continue
            else:
                local_space = img[i-K:i-K+kernel_size, j-K:j-K+kernel_size].reshape(-1)
                local_mean = np.mean(local_space)
                local_var = np.var(local_space)
                # Deal in special case (local var too small)
                if local_var < noise_var:
                    blur_img[i][j] = img[i][j]-local_var / noise_var * (img[i][j]-local_mean)
                else:
                    blur_img[i][j] = img[i][j] - noise_var / local_var  * (img[i][j] - local_mean)

    return np.uint8(blur_img)

def Recover(img, mode, noise_var=1000):
    '''
        Image recover
    '''
    if mode == 'Arithmetic':
        return myarithemeticblur(img, 7)
    elif mode == "Geometric":
        return mygeometricblur(img, 7)
    else:
        return myadaptiveblur(img, 7, noise_var)

def MyRecover(img, mode, noise_var=1000):
    '''
        GRAY and RGB seperately
    '''
    if len(img.shape) == 2:
        return Recover(img, mode, noise_var)
    else:
        blur_img = img.copy()
        for i in range(3):
            blur_img[:, :, i] = Recover(img[:, :, i], mode, noise_var)
        return blur_img

def OpencvEqualize(img, mode):
    '''
        Utilized for double check during experiment.
        Not used in final demo.
    '''
    if mode == "GRAY":
        result = cv2.equalizeHist(img)
        visual(result, "cvGRAY")
    elif mode == "RGB":
        (b, g, r) = cv2.split(img)
        bH = cv2.equalizeHist(b)
        gH = cv2.equalizeHist(g)
        rH = cv2.equalizeHist(r)
        result = cv2.merge((bH, gH, rH))
    elif mode == "HSI":
        hsi_img = RGB2HSI(img)
        (h, s, i) = hsi_img[:, :, 0], hsi_img[:, :, 1], hsi_img[:, :, 2]
        hH = h
        sH = s
        iH = cv2.equalizeHist(np.uint8(i))
        hsi_img = np.stack((hH, sH, iH), axis=2)
        result = HSI2RGB(hsi_img)

    return result

def OpencvLaplacian(img):
    '''
        Utilized for double check during experiment.
        Not used in final demo.
    '''
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    return img-laplacian, laplacian

def OpencvRecover(img, mode):
    '''
        Utilized for double check during experiment.
        Not used in final demo.
    '''
    if mode == 'Arithmetic':
        return cv2.blur(img, (7, 7))
    elif mode == 'Geometric':
        return cv2.blur(img, (7, 7))
    else:
        return cv2.blur(img, (7, 7))

if __name__ == '__main__':
    pass