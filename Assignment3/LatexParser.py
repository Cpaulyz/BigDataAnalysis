import re

input_str = r"""
N=\left[\begin{array}{cccc}
0 & \frac{1}{2} & 0 & \frac{1}{2} \\
\frac{1}{3} & 0 & 0 & \frac{1}{2} \\
\frac{1}{3} & \frac{1}{2} & 0 & 0 \\
\frac{1}{3} & 0 & 1 & 0
\end{array}\right]"""


# 对外暴露的接口
def latex_to_array(latex):
    content = get_array_str(latex)
    return matrix_to_array(content)


# 提取\begin{array}到\end{array}中间的部分
def get_array_str(latex):
    # re_array =
    pattern = re.compile(r'\\begin\{array\}\{.*\}[\w\W]*\\end\{array\}', re.DOTALL)
    # pattern = re.compile(r'[.|\n]*',re.DOTALL)
    array = pattern.findall(latex)[0]
    content = re.sub(r'(\\begin\{array\}\{.*\}(\n)*)|((\n)*\\end\{array\})', "", array)
    # print(content)
    return content


# 将latex中的数组部分转为二维数组
def matrix_to_array(content: str):
    res = []
    rows = content.split('\\\\\n')
    for row in rows:
        res.append(row_to_array(row))

    # print(res)
    return res


# 将latex中的一行转为list
def row_to_array(row: str):
    res = []
    expr_array = row.split('&')
    for expr in expr_array:
        res.append(expr_to_num(expr))
    return res


# 将latex中的一个表达式转为数字
def expr_to_num(expr: str):
    expr = expr.strip()
    if expr.isdigit():
        return expr
    else:
        matchObj = re.match(r'\\frac\{(\d+)\}\{(\d+)\}', expr)
        fenzi = int(matchObj.group(1))
        fenmu = int(matchObj.group(2))
        return fenzi / fenmu


print(latex_to_array(input_str))
