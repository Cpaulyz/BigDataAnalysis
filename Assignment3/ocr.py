import base64
import json
import requests


def encode_base64(file):
    with open(file, 'rb') as f:
        img_data = f.read()
        base64_data = base64.b64encode(img_data)
        # print(type(base64_data))
        # print(base64_data)
        # 如果想要在浏览器上访问base64格式图片，需要在前面加上：data:image/jpeg;base64,
        base64_str = str(base64_data, 'utf-8')
        # print(base64_str)
        return base64_str


#
if __name__ == '__main__':
    img_path = 'test.png'
    base64code = encode_base64(img_path)

    url = "https://www.latexlive.com:5001/api/mathpix/posttomathpix"
    # headers = {'Content-Type': 'application/json'}
    payload = {"src": "data:image/png;base64," + base64code}
    r = requests.post(url, json=payload)
    print(r.text)
    content = json.loads(r.text).get('latex_styled')
    print(content)
