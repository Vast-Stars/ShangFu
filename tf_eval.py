# -*- coding:utf-8 -*-
import tf_inference
import tf_train
import numpy
import tensorflow as tf
from PIL import Image
import Variables

def split_image(file, x, y, x_step=0, y_step=0):
    if x_step is None or x_step == 0:
        x_step = x
    if y_step is None or y_step == 0:
        y_step = y
    image = Image.open(file)
    print('Raw Image Size:', image.width, image.height)
    x_cur = 0
    Image_list = []
    x_y_list = []
    i=0
    while x_cur + x <= image.width + x_step:
        y_cur = 0
        while y_cur + y <= image.height + y_step:
            # print(x_cur,y_cur)
            img2 = image.crop((x_cur, y_cur, x_cur + x, y_cur + y))
            #img2.save('DATA/SPLIT/' + str(x_cur) + '_' + str(y_cur) + '.jpg')
            img2.save('DATA/SPLIT/' + str(i) + '.jpg')
            i+=1
            Image_list.append(img2)
            x_y_list.append(str(x_cur) + '_' + str(y_cur))
            y_cur += y_step
        x_cur += x_step
    return Image_list, x_y_list

if __name__ == '__main__':
    img_list, xy_list = split_image('DATA/001.jpg', x=128, y=128, x_step=64, y_step=64)
    imgs = [numpy.array(i) for i in img_list]
    #imgs=img_list[0].resize((128,128))

    imgs = numpy.array(imgs,dtype=numpy.float32)
    #print(imgs.shape)
    x = tf.placeholder(tf.float32,
                       [256, 128, 128, 3],
                       name='x-input')
    saver = tf.train.import_meta_graph('SAVE/model.ckpt.meta')
    # We can now access the default graph where all our metadata has been loaded
    graph = tf.get_default_graph()
    y = tf_inference.inference(x, drop_out=True, regularizer= tf.contrib.layers.l2_regularizer(Variables.REGULARIZATION_RATE))
    #train_op = graph.get_operation_by_name('loss')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('Session Begin')
        saver.restore(sess, 'SAVE/model.ckpt')

        tem=sess.run(x,feed_dict={x: imgs})
        print('INPUT SHAPE:',tem.shape)
        loss2= sess.run(y, feed_dict={x: imgs})
        print(type(loss2),loss2.shape)
        #print(loss2[50:55])
        for i in numpy.argmax(loss2,0):
            print(i,loss2[i])


