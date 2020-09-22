import re

input_str = r"""
N=\left[\begin{array}{cccc}
0 & \frac{1}{2} & 0 & \frac{1}{2} \\
\frac{1}{3} & 0 & 0 & \frac{1}{2} \\
\frac{1}{3} & \frac{1}{2} & 0 & 0 \\
\frac{1}{3} & 0 & 1 & 0
\end{array}\right]"""


# 提取\begin{array}到\end{array}中间的部分
def get_array_str(latex):
    # re_array =
    pattern = re.compile(r'\\begin\{array\}\{.*\}[\w\W]*\\end\{array\}', re.DOTALL)
    # pattern = re.compile(r'[.|\n]*',re.DOTALL)
    array = pattern.findall(latex)[0]
    content = re.sub(r'(\\begin\{array\}\{.*\}(\n)*)|((\n)*\\end\{array\})', "", array)
    print(content)
    return content


get_array_str(input_str)
