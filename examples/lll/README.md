# image-epipolar-geometry

## description
- Calculating disparity (and thus depth of the scene) using two images of a scene.
- Sort of what eyes do all the time.

## roadmap
Problems in `hw5.pdf` are solved.

## code
- All source code is in `Stereo_Reconstruction.py`.
- It reads from `left.bmp` and `right.bmp`.

## documentation
- Code is the documentation of itself.

## usage
- Use `python3 Stereo_Reconstruction.py` to generate a disparity map.
- A summary of the pipeline is given in `report.pdf`.

## demonstration
The pipeline is demonstrated below.

- Original images.

| left | right |
| --- | --- |
| ![](./left.bmp) | ![](./right.bmp) |

- SIFT matches.

| left-to-right | right-to-left |
| --- | --- |
| ![](./github/1.I1-I2.png) | ![](./github/1.I2-I1.png) |

- SIFT matches after nearest neighbour filtering.

| left-to-right | right-to-left |
| --- | --- |
| ![](./github/2.I1-I2-NN.png) | ![](./github/2.I2-I1-NN.png) |

- Bi-directional SIFT matches.

![](./github/3.I1-I2-bi.png)

- Epipolar lines.
![](./github/4.epipolar.png)

- Triangulated points with possible camera poses.
![](./github/5.point_clouds.png)

- Triangulated points with disambiguated camera pose.
![](./github/6.disambiguated_pose.png)

- Rectified images.
![](./github/7.rectified_images.png)

- Disparity map (red=nearer, blue=farther to the camera).
![](./github/8.dispariy.png)

- Disparity map vs SIFT size (red=nearer, blue=farther to the camera).
![](./github/comparision/disparity.gif)

- Disparity map over rectified left image vs SIFT size (red=nearer, blue=farther to the camera).
![](./github/comparision/blend.gif)

