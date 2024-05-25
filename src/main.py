import time
import os
import pandas as pd
import concurrent.futures

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# これでjsonモードかどうかを切り替える
is_json_mode = True


system_message = """
あなたはプロのSF小説家です。現代の期待の作家として、老若男女問わず読みやすく味のある魅力的な文章を書く作家として知られています。
ユーザーの指示に従って、小説の原稿を書いてください。

条件:
- できる限り長く詳しく書いてください。具体的には20000字程度の文章を書いてください。
- 説明的な文章となることを避け、エピソードと会話を中心に内容を表現してください。
- 会話は、哲学的な思想を含む多彩な比喩にあふれた深い含蓄のあるものにしてください。
- 情景描写は色彩豊かで、耽美的で、独創的な表現を駆使してください。
- 科学的な知識を踏まえて、舞台となる場所の情景を華麗に描写し、時代の風潮を印象付けるように解説してください。
- 初登場時に、登場人物の容貌、服装、役割と特徴を詳しく描写してください。
- 結末は余韻の残るものにしてください。
"""

if is_json_mode:
    system_message += """
    出力は以下の例のようなJSON形式で、titleキーに小説のタイトル、contentキーに小説の内容の文章を含みます。
    ```json
    {
        "title": "NEO桃太郎",
        "content": "「昔々あるところに、おじいさんとおばあさんがいました」というのは昔話であって..."
    }
    ```
    """

user_message = """
科学が高度に発展し、LLM(大規模言語モデル)のようなAI技術が日常生活に深く溶け込んだ未来に桃太郎が登場する20000字程度の日本語の長編小説を書いてください。
出力には小説の内容の文章のみを含み、そのまま出版することができる質の高い文章が期待されます。
"""


def llm_request(llm_model: str, system_message: str, user_message: str, max_tokens: int):
    client = OpenAI()

    latency_start = time.perf_counter()
    response = client.chat.completions.create(
        model=llm_model,
        max_tokens=max_tokens,
        response_format={"type": "json_object" if is_json_mode else "text"},
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    )
    latency_end = time.perf_counter()
    latency = (latency_end - latency_start) * 1000

    content = response.choices[0].message.content
    print(f"Model: {response.model}, Max tokens: {max_tokens}, Completion tokens: {
          response.usage.completion_tokens}, Latency: {latency:.2f} ms, json?: {is_json_mode}")

    return response.model, response.usage.completion_tokens, latency, content


if __name__ == "__main__":
    label, x, y, json, contents = [], [], [], [], []

    # 並列実行
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for _ in range(5):
            for max_tokens in range(100, 4096, 200):
                for model in ["gpt-3.5-turbo", "gpt-4o"]:
                    futures.append(executor.submit(
                        llm_request, model, system_message, user_message, max_tokens))
        for future in concurrent.futures.as_completed(futures):
            response_model, completion_tokens, latency, content = future.result()
            # label, x, y を保存
            label.append(response_model)
            x.append(completion_tokens)
            y.append(latency)
            json.append(is_json_mode)
            contents.append(content)

    df = pd.DataFrame({"label": label, "completion_tokens": x,
                      "latency": y, "is_json_mode": json, "content": contents})

    output_dir = "./out"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df.to_csv(f"{output_dir}/results_{time.strftime("%Y%m%d_%H%M%S",
              time.localtime())}.csv", index=False)
