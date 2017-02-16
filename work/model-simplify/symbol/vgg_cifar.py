"""References:

http://torch.ch/blog/2015/07/30/cifar.html
"""
import mxnet as mx

def get_symbol(num_classes):
    ## define alexnet
    data = mx.symbol.Variable(name="data")
    # group 1
    conv1_1 = mx.symbol.Convolution(data=data,
                                    num_filter=64,
                                    kernel=(3, 3),
                                    stride=(1,1),
                                    pad=(1, 1),
                                    name="conv1_1")
    bn1_1 = mx.symbol.BatchNorm(data=conv1_1, eps=1e-3, name='bn1_1')
    relu1_1 = mx.symbol.Activation(data=bn1_1, act_type="relu", name="relu1_1")
    drop1_1 = mx.symbol.Dropout(data=relu1_1, p=0.3, name="drop1_1")
    conv1_2 = mx.symbol.Convolution(data=drop1_1,
                                    num_filter=64,
                                    kernel=(3, 3),
                                    stride=(1,1),
                                    pad=(1, 1),
                                    name="conv1_2")
    bn1_2 = mx.symbol.BatchNorm(data=conv1_2, eps=1e-3, name='bn1_2')
    relu1_2 = mx.symbol.Activation(data=bn1_2, act_type="relu", name="relu1_2")
    pool1 = mx.symbol.Pooling(data=relu1_2, pool_type="max", kernel=(2, 2),
                              stride=(2,2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(data=pool1,
                                    num_filter=128,
                                    kernel=(3, 3),
                                    stride=(1,1),
                                    pad=(1, 1),
                                    name="conv2_1")
    bn2_1 = mx.symbol.BatchNorm(data=conv2_1, eps=1e-3, name='bn2_1')
    relu2_1 = mx.symbol.Activation(data=bn2_1, act_type="relu", name="relu2_1")
    drop2_1 = mx.symbol.Dropout(data=relu2_1, p=0.4, name="drop2_1")
    conv2_2 = mx.symbol.Convolution(data=drop2_1,
                                    num_filter=128,
                                    kernel=(3, 3),
                                    stride=(1,1),
                                    pad=(1, 1),
                                    name="conv2_2")
    bn2_2 = mx.symbol.BatchNorm(data=conv2_2, eps=1e-3, name='bn2_2')
    relu2_2 = mx.symbol.Activation(data=bn2_2, act_type="relu", name="relu2_2")
    pool2 = mx.symbol.Pooling(data=relu2_2, pool_type="max", kernel=(2, 2),
                              stride=(2,2), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(data=pool2,
                                    num_filter=256,
                                    kernel=(3, 3),
                                    stride=(1,1),
                                    pad=(1, 1),
                                    name="conv3_1")
    bn3_1 = mx.symbol.BatchNorm(data=conv3_1, eps=1e-3, name='bn3_1')
    relu3_1 = mx.symbol.Activation(data=bn3_1, act_type="relu", name="relu3_1")
    drop3_1 = mx.symbol.Dropout(data=relu3_1, p=0.4, name="drop3_1")
    conv3_2 = mx.symbol.Convolution(data=drop3_1,
                                    num_filter=256,
                                    kernel=(3, 3),
                                    stride=(1,1),
                                    pad=(1, 1),
                                    name="conv3_2")
    bn3_2 = mx.symbol.BatchNorm(data=conv3_2, eps=1e-3, name='bn3_2')
    relu3_2 = mx.symbol.Activation(data=bn3_2, act_type="relu", name="relu3_2")
    drop3_2 = mx.symbol.Dropout(data=relu3_2, p=0.4, name="drop3_2")
    conv3_3 = mx.symbol.Convolution(data=drop3_2,
                                    num_filter=256,
                                    kernel=(3, 3),
                                    stride=(1,1),
                                    pad=(1, 1),
                                    name="conv3_3")
    bn3_3 = mx.symbol.BatchNorm(data=conv3_3, eps=1e-3, name='bn3_3')
    relu3_3 = mx.symbol.Activation(data=bn3_3, act_type="relu", name="relu3_3")
    pool3 = mx.symbol.Pooling(data=relu3_3, pool_type="max", kernel=(2, 2),
                              stride=(2,2), name="pool3")
    # group 4
    conv4_1 = mx.symbol.Convolution(data=pool3,
                                    num_filter=512,
                                    kernel=(3, 3),
                                    stride=(1,1),
                                    pad=(1, 1),
                                    name="conv4_1")
    bn4_1 = mx.symbol.BatchNorm(data=conv4_1, eps=1e-3, name='bn4_1')
    relu4_1 = mx.symbol.Activation(data=bn4_1, act_type="relu", name="relu4_1")
    drop4_1 = mx.symbol.Dropout(data=relu4_1, p=0.4, name="drop4_1")
    conv4_2 = mx.symbol.Convolution(data=drop4_1,
                                    num_filter=512,
                                    kernel=(3, 3),
                                    stride=(1,1),
                                    pad=(1, 1),
                                    name="conv4_2")
    bn4_2 = mx.symbol.BatchNorm(data=conv4_2, eps=1e-3, name='bn4_2')
    relu4_2 = mx.symbol.Activation(data=bn4_2, act_type="relu", name="relu4_2")
    drop4_2 = mx.symbol.Dropout(data=relu4_2, p=0.4, name="drop4_2")
    conv4_3 = mx.symbol.Convolution(data=drop4_2,
                                    num_filter=512,
                                    kernel=(3, 3),
                                    stride=(1,1),
                                    pad=(1, 1),
                                    name="conv4_3")
    bn4_3 = mx.symbol.BatchNorm(data=conv4_3, eps=1e-3, name='bn4_3')
    relu4_3 = mx.symbol.Activation(data=bn4_3, act_type="relu", name="relu4_3")
    pool4 = mx.symbol.Pooling(data=relu4_3, pool_type="max", kernel=(2, 2),
                              stride=(2,2), name="pool4")
    # group 5
    conv5_1 = mx.symbol.Convolution(data=pool4,
                                    num_filter=512,
                                    kernel=(3, 3),
                                    stride=(1,1),
                                    pad=(1, 1),
                                    name="conv5_1")
    bn5_1 = mx.symbol.BatchNorm(data=conv5_1, eps=1e-3, name='bn5_1')
    relu5_1 = mx.symbol.Activation(data=bn5_1, act_type="relu", name="relu5_1")
    drop5_1 = mx.symbol.Dropout(data=relu5_1, p=0.4, name="drop5_1")
    conv5_2 = mx.symbol.Convolution(data=drop5_1,
                                    num_filter=512,
                                    kernel=(3, 3),
                                    stride=(1,1),
                                    pad=(1, 1),
                                    name="conv5_2")
    bn5_2 = mx.symbol.BatchNorm(data=conv5_2, eps=1e-3, name='bn5_2')
    relu5_2 = mx.symbol.Activation(data=bn5_2, act_type="relu", name="relu5_2")
    drop5_2 = mx.symbol.Dropout(data=relu5_2, p=0.4, name="drop5_2")
    conv5_3 = mx.symbol.Convolution(data=drop5_2,
                                    num_filter=512,
                                    kernel=(3, 3),
                                    stride=(1,1),
                                    pad=(1, 1),
                                    name="conv5_3")
    bn5_3 = mx.symbol.BatchNorm(data=conv5_3, eps=1e-3, name='bn5_3')
    relu5_3 = mx.symbol.Activation(data=bn5_3, act_type="relu", name="relu5_3")
    pool5 = mx.symbol.Pooling(data=relu5_3, pool_type="max", kernel=(2, 2),
                              stride=(2,2), name="pool5")
    # group 6
    flatten = mx.symbol.Flatten(data=pool5, name="flatten")
    drop6 = mx.symbol.Dropout(data=flatten, p=0.5, name="drop6")
    fc6 = mx.symbol.FullyConnected(data=drop6, num_hidden=512, name="fc6")
    bn6 = mx.symbol.BatchNorm(data=fc6, eps=1e-3, name='bn6')
    relu6 = mx.symbol.Activation(data=bn6, act_type="relu", name="relu6")
    # output
    drop7 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop7")
    fc7 = mx.symbol.FullyConnected(data=drop7, num_hidden=num_classes, name="fc7")
    softmax = mx.symbol.SoftmaxOutput(data=fc7, name='softmax')
    return softmax
    #return mx.symbol.Group([flatten,relu6,softmax])
