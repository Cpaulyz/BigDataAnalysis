from LatexParser import latex_to_array
from OCR import get_latex_from_file

input_str = r"""
N=\left[\begin{array}{cccc}
0 & \frac{1}{2} & 0 & \frac{1}{2} \\
\frac{1}{3} & 0 & 0 & \frac{1}{2} \\
\frac{1}{3} & \frac{1}{2} & 0 & 0 \\
\frac{1}{3} & 0 & 1 & 0
\end{array}\right]"""


# array: 转移矩阵(二维数组)
# time:迭代次数
# d: 阻尼系数
def calculate(array, time=10, d=0.85):
    length = len(array)
    value = [1 / length] * length  # page rank value list
    print('init', value)
    for i in range(time):
        new_value = []  # new page rank value list
        for row_index in range(length):
            row = array[row_index]
            tmp = (1 - d) / length  # tmp is new value item
            for col_index in range(len(row)):
                tmp += d * float(row[col_index]) * value[col_index]
            new_value.append(tmp)
        print('iter', i + 1, new_value)
        value = new_value


# 传入文件路径列表
def page_rank(paths):
    for img_path in paths:
        latex = get_latex_from_file('test.png')
        arr = latex_to_array(latex)
        # calculate(arr, 10, 1)  # 不考虑阻尼系数
        calculate(arr, 10)  # 考虑阻尼系数


if __name__ == '__main__':
    latex = get_latex_from_file('matrix.png')
    # latex = input_str # 本次作业获取的input_str，直接用的话不需要请求时间
    arr = latex_to_array(latex)
    # calculate(arr, 10, 1)  # 不考虑阻尼系数
    calculate(arr, 10)  # 考虑阻尼系数
