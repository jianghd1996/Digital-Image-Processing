# web-app for API image manipulation

from flask import Flask, request, render_template, send_from_directory
import os
from PIL import Image
#从文件夹algorithm引入算法接口（Morphological_Transformation为形态学处理的代码）
from algorithm import Morphological_Transformation
import numpy as np
import cv2
import time

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# default access page
@app.route("/")
def main():
    return render_template('new_index.html')


# upload selected image and forward to processing page
@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'static/temp_images/')

    # create image directory if not found
    if not os.path.isdir(target):
        os.mkdir(target)

    # retrieve file from html file-picker
    upload = request.files.getlist("file")[0]
    print("File name: {}".format(upload.filename))
    filename = upload.filename

    # file support verification
    ext = os.path.splitext(filename)[1]
    if ext in [".jpg", ".png", ".tif"]:
        print("File accepted")
    else:
        return render_template("new_error.html", message="The selected file is not supported"), 400

    if '.tif' in filename:
        filename = filename.replace(".tif", '.jpg')

    # save file
    local_time = time.localtime(time.time())

    filename= "filename-{}-{}".format(local_time,filename)

    destination = "/".join([target, filename])
    if os.path.isfile(destination):
        os.remove(destination)
    print("File saved to:", destination)
    upload.save(destination)

    img = cv2.imread(destination, 1)
    if (img[:, :, 0] == img[:, :, 1]).all() and (img[:, :, 1] == img[:, :, 2]).all() and (img[:, :, 0] == img[:, :, 2]).all():
        img = cv2.imread(destination, 0)
    cv2.imwrite(destination, img)

    # forward to processing page
    return render_template("new_processing.html", image_name=filename)

@app.route("/M_T", methods=["POST"])
def M_T():

    # retrieve parameters from html form
    mode = request.form['mode']
    filename = request.form['image']

    if '.tif' in filename:
        filename = filename.replace(".tif", '.jpg')

    target = os.path.join(APP_ROOT, 'static/temp_images')
    destination = "/".join([target, filename])

    img = cv2.imread(destination, 1)
    if (img[:, :, 0] == img[:, :, 1]).all() and (img[:, :, 1] == img[:, :, 2]).all() and (img[:, :, 0] == img[:, :, 2]).all():
        img = cv2.imread(destination, 0)

    raw_name = "raw-{}-{}".format(mode, filename)
    result_name = "result-{}-{}".format(mode, filename)

    # check mode
    if mode in ['GRAY', 'RGB', 'HSI']:
        destination = "/".join([target, raw_name])
        cv2.imwrite(destination, img)

        myimg = Morphological_Transformation.MyEqualize(img, mode)
        destination = "/".join([target, result_name])
        cv2.imwrite(destination, myimg)

        raw_hist = cv2.imread("raw_hist.png")
        result_hist = cv2.imread("my_hist.png")

        raw_hist_name = "raw-hist-{}-{}".format(mode, filename)
        destination = "/".join([target, raw_hist_name])
        cv2.imwrite(destination, raw_hist)

        result_hist_name = "my-hist-{}-{}".format(mode, filename)
        destination = "/".join([target, result_hist_name])
        cv2.imwrite(destination, result_hist)

        return render_template("MT_1.html", raw_name=raw_name, result_name=result_name,
                               raw_hist_name=raw_hist_name, result_hist_name=result_hist_name)

    elif mode in ['Laplacian_1', 'Laplacian_2']:
        myimg, laplace_img, norm_img = Morphological_Transformation.MyLaplacian(img, mode)
        destination = "/".join([target, raw_name])
        cv2.imwrite(destination, img)
        destination = "/".join([target, result_name])
        cv2.imwrite(destination, myimg)

        laplace_img_name = "laplace-hist-{}-{}".format(mode, filename)
        norm_img_name = "norm-hist-{}-{}".format(mode, filename)

        destination = "/".join([target, laplace_img_name])
        cv2.imwrite(destination, laplace_img)
        destination = "/".join([target, norm_img_name])
        cv2.imwrite(destination, norm_img)

        return render_template("MT_2.html", raw_name=raw_name, result_name=result_name,
                               norm_name=norm_img_name, laplace_name=laplace_img_name)

    elif mode in ['Arithmetic', 'Geometric', 'Adaptive']:
        noise_img, noise_size = Morphological_Transformation.add_noise(img.copy(), "gaussian")
        myimg = Morphological_Transformation.MyRecover(noise_img, mode, noise_size)

        destination = "/".join([target, raw_name])
        cv2.imwrite(destination, img)
        destination = "/".join([target, result_name])
        cv2.imwrite(destination, myimg)

        noise_img_name = "noise-{}-{}".format(mode, filename)
        destination = "/".join([target, noise_img_name])
        cv2.imwrite(destination, noise_img)

        return render_template("MT_3.html", raw_name=raw_name, result_name=result_name,
                               noise_name= noise_img_name)

    return render_template("new_error.html", message="Invalid mode (vertical or horizontal)"), 400

# retrieve file from 'static/temp_images' directory
@app.route('/static/temp_images/<filename>')
def send_image(filename):
    return send_from_directory("static/temp_images", filename)


if __name__ == "__main__":
    app.run(debug = True)