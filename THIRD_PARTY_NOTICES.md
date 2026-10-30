# 第三方资源声明

本项目代码以 [MIT](LICENSE) 许可发布。下列第三方资源**不随本仓库分发**，由
`python -m aidetect fetch` 在用户本机从 Hugging Face / ModelScope 下载（下载地址见
[README §二](README.md#资源下载地址大文件不随仓库分发)）。它们的许可与引用要求独立于本项目的
MIT 许可，分发或再发布时请分别遵循，**使用前请以官方模型 / 数据集卡片为准**。

| 资源 | 用途 | 上游 | 许可 | 体积 |
| --- | --- | --- | --- | --- |
| `yuchuantian/AIGC_detector_zhv3` | AI 文本判别器 v3（站点对照轨；即 cmj 的上游原版） | <https://huggingface.co/yuchuantian/AIGC_detector_zhv3> | Apache-2.0 | 390.2 MB |
| `mokawa3018/cmj-chinese-aigc-text-detector` | AI 文本判别器 v2（首屏疑似度） | <https://huggingface.co/mokawa3018/cmj-chinese-aigc-text-detector> ／ <https://modelscope.cn/models/mokawa3018/cmj-chinese-aigc-text-detector> | Apache-2.0 | 390.2 MB |
| `Hello-SimpleAI/chatgpt-detector-roberta-chinese` | AI 文本判别器 v1（保守口径） | <https://huggingface.co/Hello-SimpleAI/chatgpt-detector-roberta-chinese> | Apache-2.0 | 390.2 MB |
| `uer/gpt2-chinese-cluecorpussmall` | 困惑度 / GLTR 引擎 | <https://huggingface.co/uer/gpt2-chinese-cluecorpussmall> | 以模型卡片为准 | 401.4 MB |
| `Hello-SimpleAI/HC3-Chinese` | 标定与域外评测语料 | <https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese> | 以数据集卡片为准 | 20.7 MB |
| `Qwen/Qwen2.5-0.5B`（可选） | 现代中文 LM 实验组件，当前主链路不使用 | <https://huggingface.co/Qwen/Qwen2.5-0.5B> ／ <https://modelscope.cn/models/Qwen/Qwen2.5-0.5B> | Apache-2.0 | 约 950 MB |

## 方法与思路的出处

- HC3 数据集的人机对照构造方式（项目内标定 / 测试协议据此设计）；
- `cmj-chinese-aigc-text-detector` 由 `yuchuantian/AIGC_detector_zhv3` 微调而来（其基座为
  `hfl/chinese-roberta-wwm-ext`），本项目只把它们当作**预训练权重**使用；
- GLTR 的 token 分桶（top-10 / top-100 / tail）思路，用于困惑度引擎的特征；
- 温度缩放（temperature scaling）的概率校准做法，用于融合后概率的校准；
- 「站点对照轨道」的档位口径（高 70 / 中 60 / 低 50 分、疑似文字占比）来自商业 AIGC 检测
  报告（PaperPass 免费版）的**输出格式**，并用该报告在锚点文档上的公开数字反解档位线；
  该轨道不包含、也不复制任何商业站点的模型或代码，只是让本地结果与站点数字可对照。

以上均为方法层面的借鉴，本项目自行实现，未复制第三方代码；已知同类开源项目
（如 `Hello-SimpleAI/chatgpt-detector`、`Gx664/AIGC-detector`）的代码与许可未被使用，
本项目为独立实现。
