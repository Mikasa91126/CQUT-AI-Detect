# 第三方资源声明

本项目代码以 [MIT](LICENSE) 许可发布。下列第三方资源**不随本仓库分发**，由
`python -m aidetect fetch` 在用户本机从 Hugging Face 下载（下载地址见
[README §二](README.md#资源下载地址大文件不随仓库分发)）。它们的许可与引用要求独立于本项目的
MIT 许可，分发或再发布时请分别遵循，**使用前请以官方模型 / 数据集卡片为准**。

| 资源 | 用途 | 上游 | 许可 | 体积 |
| --- | --- | --- | --- | --- |
| `Hello-SimpleAI/chatgpt-detector-roberta-chinese` | AI 文本判别器（主引擎） | <https://huggingface.co/Hello-SimpleAI/chatgpt-detector-roberta-chinese> | Apache-2.0 | 390.2 MB |
| `uer/gpt2-chinese-cluecorpussmall` | 困惑度 / GLTR 引擎 | <https://huggingface.co/uer/gpt2-chinese-cluecorpussmall> | 以模型卡片为准 | 401.4 MB |
| `Hello-SimpleAI/HC3-Chinese` | 标定与域外评测语料 | <https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese> | 以数据集卡片为准 | 20.7 MB |

## 方法与思路的出处

- HC3 数据集的人机对照构造方式（项目内标定 / 测试协议据此设计）；
- GLTR 的 token 分桶（top-10 / top-100 / tail）思路，用于困惑度引擎的特征；
- 温度缩放（temperature scaling）的概率校准做法，用于融合后概率的校准。

以上均为方法层面的借鉴，本项目自行实现，未复制第三方代码；已知同类开源项目
（如 `Hello-SimpleAI/chatgpt-detector`、`Gx664/AIGC-detector`）的代码与许可未被使用，
本项目为独立实现。
