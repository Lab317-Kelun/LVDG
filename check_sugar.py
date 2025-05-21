import requests
import openai
import base64
import cv2


# 获取图像帧与相机参数
def get_image_from_camera():
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 捕获帧
    ret, frame = cap.read()

    if not ret:
        print("无法接收帧")
        return None

    # 保存拍摄的图像
    initial_image_path = "initial_image.jpg"
    cv2.imwrite(initial_image_path, frame)
    print("图像已保存到", initial_image_path)
    return initial_image_path



def upload_image_to_imgbb(image_path):
    # jpg_to_url api_key
    api_key = "81b2d79cc6ae2fbac18ec8c011875295"

    url = "https://api.imgbb.com/1/upload"

    with open(image_path, "rb") as image_file:
        # 读取图像并转换为base64编码
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

        payload = {
            "key": api_key,
            "image": image_base64,
        }

        response = requests.post(url, data=payload)

        # print("Request URL:", response.url)
        # print("Request Payload:", payload)
        # print("Response Status Code:", response.status_code)
        # print("Response Content:", response.content)

        if response.status_code == 200:
            data = response.json()
            if data["status"] == 200:
                return data["data"]["url"]
            else:
                raise ValueError("Image upload failed: " + data.get("error", {}).get("message", "Unknown error"))
        else:
            raise ValueError("HTTP request failed with status code " + str(response.status_code))


def main():
    image_path = get_image_from_camera()
    image_url = upload_image_to_imgbb(image_path)
    print("Image URL:", image_url)

    # OpenAI API 密钥
    openai.api_key = 'sk-svcacct-adsco1VL1C_E1r4FeXD3FXpsx9qquJ1al827Qr7pDwfi6-FsENdzmteqiaaAi4kgT3BlbkFJhtKk9DuAKnFTfqrP6Vv6BXdMQfM0YoHW4DMVpUnWk2vKaggTgbN_b68-mTVfZR8A'
    openai.api_base = "https://api.openai-proxy.com/v1"

    # 调用 OpenAI API 进行聊天生成
    response1 = openai.ChatCompletion.create(
        model="gpt-4o",  # 使用 GPT-4 模型（请根据权限调整模型名称）
        messages=[
            {
                "role": "user",
                "content": [
                    # 提示词
                    {"type": "text", "text": "Is it highly likely that the yellow spoon contains white sugar?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    },
                ],
            }
        ],
    )

    # 打印响应
    print(response1['choices'][0]['message']['content'])

    response2 = openai.ChatCompletion.create(
        model="gpt-4o",  # 使用 GPT-4o 模型
        messages=[
            {
                "role": "user",
                "content": [
                    # 改提示词
                    {"type": "text", "text": response1['choices'][0]['message']['content']+'Please analyze and understand the previous sentence to see if the spoon is likely to contain sugar. If so, please reply with 1. If not, please reply with 0'}
                ],
            }
        ],
    )

    # 最终输出1或者0 1为勺子中含有糖 0为勺子中不含有糖
    response = response2['choices'][0]['message']['content']

    print(response)
    return response

if __name__ == "__main__":
    main()
