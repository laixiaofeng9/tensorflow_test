import numpy as np


# 1 定义损失函数
def compute_error_for_line_given_pionts(b, w, points):
    """
    计算损失函数
    :param b:
    :param w:
    :param points:
    :return:
    """
    totalError = 0
    for i in range(len(points)):
        x = points[i][0]
        y = points[i][1]
        # 计算均方误差
        error = ((w*x + b) - y) ** 2
        totalError += error

    # 对损失函数求均值
    vag_error = totalError / float(len(points))
    return vag_error


# 2 计算梯度更新函数
def step_gradient(b_current, w_current, points, learningRate):
    """
    计算更新函数
    :param b_current:
    :param w_current:
    :param points:
    :param learningRate:
    :return:
    """
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(len(points)):
        x = points[i][0]
        y = points[i][1]
        # grad_b = 2(wx+b-y)
        b_gradient += (2/N) * ((w_current * x + b_current) - y)
        # grad_w = 2(wx+b-y) * x
        w_gradient += (2/N) * ((w_current * x + b_current) - y) * x

    new_b = b_current - learningRate * b_gradient
    new_w = w_current - learningRate * w_gradient

    return new_b, new_w


# 3 循环迭代的次数
def gradient_descent_runner(points, starting_b, starting_w, learning_rate, num_iterations):
    """
    迭代计算
    :param points:
    :param starting_b:
    :param starting_w:
    :param learning_rate:
    :param num_iterations:
    :return:
    """
    b = starting_b
    w = starting_w
    # 迭代的次数
    for i in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), learning_rate)
        if i % 50 == 0:
            loss = compute_error_for_line_given_pionts(b, w, points)
            print("迭代{}次, b:{}, w:{}, loss:{}".format(i, b, w, loss))
    return b, w


def make_data():
    """
    生产数据
    :return:
    """
    data = []
    n = 100
    for i in range(n):
        x = np.random.uniform(-10.0, 10.0)
        eps = np.random.normal(0., 0.1)
        y = 5.123*x + 9.45 + eps
        data.append([x, y])
    return data


def main():
    """
    主函数
    :return:
    """
    learning_rate = 0.01
    initial_b = 0
    initial_w = 0
    num_iterations = 2000
    data = make_data()
    b, w = gradient_descent_runner(data, initial_b, initial_w, learning_rate, num_iterations)
    loss = compute_error_for_line_given_pionts(b, w, data)
    print("训练完后, b:{}, w:{}, loss:{}".format(b, w, loss))
    return None


if __name__ == '__main__':
    main()

