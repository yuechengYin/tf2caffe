# tf2caffe
此脚本暂时只支持卷积层、pooling层、fc层的转换。
## 读取tf的模型权重，写入.prototxt文件内
- 写一个模型部署文件（.prototxt)。
- copy一份作为权重填充模板,用 `python t.py `将tf模型的权重取出来并写入 `.prototxt `文件中。
- tf输入的特征图是N*H*W*C(和原图像是一样的），caffe的输入特征图是N*C*H*W。
- tf中的卷积核是H*W*Cin*Cout，caffe的卷积核是Cout*Cin*H*W.
## 将填充权值以后的.prototxt文件写成二进制.caffemodel文件
- 将`CmakeLists.txt`和`wm.cpp`置于同一个路径下。
- `cmake . `
- `make`
- 运行 `./write_model` 后就会生成`.caffemodel`文件。


