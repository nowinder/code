import tensorflow as tf
# import os
# print(os.getcwd())
# import sys
# sys.path.append('G:/Zheng_caizhi/Pycharmprojects/MLP-Mixer/nd_mlp_mixer')
# from nd_mlp_mixer import dense_on_axis
# import nd_mlp_mixer.dense_on_axis as dense_on_axis
import dense_on_axis
'''vscode运行报错：
Traceback (most recent call last):
  File "g:\Zheng_caizhi\Pycharmprojects\MLP-Mixer\nd_mlp_mixer\tests\test_dense_on_axis.py", line 3, in <module>
    from nd_mlp_mixer import dense_on_axis
ModuleNotFoundError: No module named 'nd_mlp_mixer'
一直没法解决，测试时可以拖到同一文件夹下，正常'''
def test_DenseOnAxis():
    "Tests that the shapes are as expected."
    X = tf.random.uniform([10, 10, 10, 2])
    layer = dense_on_axis.DenseOnAxis(5, axis=1)
    result = layer(X)
    assert result.shape == (10, 5, 10, 2), f"Unexpected output size, {result.shape}."


def test_linear_on_axis():
    "Tests that the shapes are as expected."
    X = tf.random.uniform([10, 10, 10, 2])
    weights = tf.random.uniform([10, 5])
    bias = tf.random.uniform([1, 5, 1, 1])
    axis = 1
    result = dense_on_axis.linear_on_axis(X, weights, bias, axis)
    assert result.shape == (10, 5, 10, 2), f"Unexpected output size, {result.shape}."

test_DenseOnAxis()
test_linear_on_axis()