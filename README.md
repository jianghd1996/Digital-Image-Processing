Digital-Image-Processing

This repo is used to learn the course digital image processing, I will try to realize different image processing algorithm, and make a comparison with standard Library. 

While the code is realized in very inefficient way under python, it may run very slow. Further optimization on calculation could be modified.



### Histogram equalization

---

The histogram, describes the distribution of pixels in an image. Use $h(r_k)$ as the number of pixels with gray value $r_k$, the normalized distribution function $p(r_k)=\frac{h(r_k)}{n}$. We hope to find a mapping function $T(r)$ to map $p(r_k)$ to $s(r)$ so that, 

- the relative partial order before and after $T(r)$ is the same
- the distribution of $s(r)$ is approximate an uniform distribution

A valid mapping function is $s(r)=T(r)=\int_{0}^{r}p_r(w)dw$, which meet
$$
\begin{align}
	p_s(s) &= p_r(r)|\frac{dr}{ds}|, \\
	\frac{ds}{dr} &= \frac{d{T(r)}}{dr}=\frac{d}{dr}[\int_{0}^{r}p_r(w)dw]=p_r(r), \\
	p_s(s)&=p_r(r)|\frac{dr}{ds}|=p_r(r)|\frac{1}{p_r(r)}|=1.
\end{align}
$$
usually, we use discrete form :
$$
s_k=T(r_k)=\sum_{j=0}^{k}p_r(r_j)=\sum_{j=0}^{k}\frac{n_j}{n}
$$
I realize this operation both in gray, RGB and HSI images. 

Results :

Gray images (raw, raw histogram, equalized, equalized histogram)

![Histogram_equalize_gray](https://github.com/jianghd1996/Digital-Image-Processing/tree/main/result/Histogram_equalize_gray.png)

Color (raw, RGB, HSI)

![Histogram_equalize_color](https://github.com/jianghd1996/Digital-Image-Processing/tree/main/result/Histogram_equalize_color.png)



### Laplacian image sharpening

---

Laplacian operator is used to extract the sharpen part of an image, and 
$$
\begin{align}
	\nabla^2{f}&=\frac{\partial^2 f}{\partial x^2}+\frac{\partial^2 f}{\partial y^2} \\
	\frac{\partial^2 f}{\partial x^2} &= f(x+1, y)+f(x-1,y)-2f(x,y) \\
	\frac{\partial^2 f}{\partial y^2} &= f(x, y+1)+f(x,y-1)-2f(x,y) \\
	\nabla^2{f}&=[f(x+1,y)+f(x-1,y)+f(x,y+1)+f(x,y-1)]-4f(x,y).
\end{align}
$$
Then, substract the raw image with the laplacian operator,

$g(x,y)=f(x,y)-\nabla^2 f(x,y)$

Also, other laplacian operator could be used,
$$
\begin{align}
\nabla^2{f}&=[f(x+1,y)+f(x-1,y)+f(x,y+1)+f(x,y-1) \\
		   &+f(x+1, y+1)+f(x+1,y-1)+f(x-1,y+1)+f(x-1,y-1)]-8f(x,y).
\end{align}
$$


Result (raw, laplacian, normalized laplacian, sharpen)![Laplacian_gray](https://github.com/jianghd1996/Digital-Image-Processing/tree/main/result/Laplacian_gray.png)

![Laplacian_color](https://github.com/jianghd1996/Digital-Image-Processing/tree/main/result/Laplacian_color.png)



### FFT transformation

---



### Correlation matching

---



### Gaussian Low Pass (GLPF)

---



### Image restore with spatial filters

---

Different noise function :



Image restoration :

The principle is to reduce the noise by mathematical functions. With different noise, different filters will have different performance. 

There is no filter performs best for all kinds of noises.



Mean filter :

Mean filter is good for gaussian noise.

Arithmetic mean filter and geometric mean filter
$$
\begin{align}
	\hat{f}(x, y)&=\frac{1}{mn}\sum_{(s,t)\in S_{xy}}g(s,t), \\
	\hat{f}(x, y)&=(\Pi_{(s,t)\in S_{xy}}g(s,t))^{\frac{1}{mn}}.
\end{align}
$$
Adaptive mean filter. It requires known the noise size, and use different weight of filter according to local noise variance towards global noise variance.

Define $\sigma_{\eta}^2, \sigma_{L}^2$ as global variance and local variance, $m_L$ as local mean pixel value. The filter perform as,
$$
\hat{f}(x,y)=g(x,y)-\frac{\sigma_{\eta}^2}{\sigma_{L}^2}(g(x,y)-m_L),
$$
in practice, I find when $\sigma_{\eta}^2>\sigma_{L}^2$, it is better to use,
$$
\hat{f}(x,y)=g(x,y)-\frac{\sigma_{L}^2}{\sigma_{\eta}^2}(g(x,y)-m_L).
$$
Result (raw, noise, arithmetic, geometric, adaptive)

![Spacial_image_restore_mean_filter](https://github.com/jianghd1996/Digital-Image-Processing/tree/main/result/Spacial_image_restore_mean_filter_gray.png)

![Spacial_image_restore_mean_filter_color](https://github.com/jianghd1996/Digital-Image-Processing/tree/main/result/Spacial_image_restore_mean_filter_color.png)



### DCT transformation

---



### Morphological transformation

---



### Edge detection and connection

---



