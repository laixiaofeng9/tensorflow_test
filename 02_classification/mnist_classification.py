import tensorflow as tf
from tensorflow import keras
import os
import time

start_time = time.time()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 导入mnist数据集
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print("dataset train:", x_train.shape, y_train.shape)
print("dataset test:", x_test.shape, y_test.shape)

# 由于导入的数据集是numpy格式，为了使用tensorflow的并行运算
# 将其转换为tensor格式
x_train= tf.convert_to_tensor(x_train, dtype = tf.float32)/255.
y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
y = tf.one_hot(y_train, depth=10)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y))
# 每次加载200张的图片
train_dataset = train_dataset.batch(200)

# print("train", train_dataset.shape)
print("label", y.shape)


# 准备网络结构
model = keras.Sequential([
    keras.layers.Dense(512, activation="relu"),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(10)
])

# 分批次梯度下降法，每个批次用200条数据
optimizer = keras.optimizers.SGD(learning_rate=0.001)


# 一个epoch就是对数据集循环迭代一次、
# 一个step就是对一个batch循环一次
def train_epoch(epoch):
    """
    每次训练一次所有的数据
    :param epoch:
    :return:
    """
    # 这个for将循环60000/200=300次, 利用tensorflow的分布式训练
    for step, (x, y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # [b, 28, 28] -> [b, 784]
            # 将二维数组打平为一维数组
            x = tf.reshape(x, (-1, 28*28))

            # 模型已经定义号了，直接传入x数据后
            # 得到[b, 10] 维的数组数据
            out = model(x)
            # 计算loss函数
            loss = tf.reduce_sum(tf.square(out - y)) / float(x.shape[0])

        # step3 优化函数的更新 w1 w2 w3 b1 b2 b3
        grads = tape.gradient(loss, model.trainable_variables)

        # w` = w - lr * grads 优化函数更新
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # 打印每一百步的loss
        if step % 100 == 0:
            print(epoch, step, loss.numpy())


def main():
    # 对数据集迭代30次, 及模型训练的次数，用于更新参数
    for epoch in range(30):
        train_epoch(epoch)


if __name__ == '__main__':
    main()
    print(model.summary())
    print("total time {} s".format(time.time() - start_time))
