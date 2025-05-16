---
title: 魔搭社区和HuggingFace使用攻略
date: 2025-05-16 10:00:00
tags:
  - 炼丹小技巧
  - 魔搭
  - HuggingFace
categories:
  - 炼丹
---
# 魔搭社区和HuggingFace使用攻略

## 魔搭社区

```
#安装环境
pip install -r requirements

# 创建模型文件夹
# mkdir 文件名
mkdir DeepSeek-R1-Distill-Qwen-7B

# 从modelscope拉取模型文件到文件夹
# modelscope download --model 【modelscope的模型名，按照下图的步骤获取】 --local_dir ./上一步创建的文件夹名字
modelscope download --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --local_dir ./DeepSeek-R1-Distill-Qwen-7B

#数据集下载
# mkdir 【数据集名字】
mkdir dataset

# modelscope download --dataset 【'modelscope中数据集的id，如下图'】 --local_dir './dataset'
modelscope download --dataset 'FreedomIntelligence/medical-o1-reasoning-SFT' --local_dir './dataset'

```

### 获取模型

![image-20250516185241883](https://fantasticnana.xyz/2025/05/cdb86874de15b92a18b0765a51a7c5a4.png)

### 获取数据集

![image-20250516185301257](https://fantasticnana.xyz/2025/05/4cf33db05d3f0d735605b7fe1d2e00af.png)

## Hugging Face

```
# 或者使用国内镜像加速
pip install datasets -U -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install huggingface_hub[cli] -U -i https://pypi.tuna.tsinghua.edu.cn/simple


# 重要！！如果访问 huggingface.co 速度慢，尝试设置国内镜像端点
export HF_ENDPOINT="https://hf-mirror.com"
#验证环境变量是否设置成功 (可选)
echo $HF_ENDPOINT


mkdir ruozhiba
huggingface-cli download hfl/ruozhiba_gpt4 --repo-type dataset --local-dir ./ruozhiba

hfl/ruozhiba_gpt替换为你想要的模型id
ruozhiba替换为你的下载路径
```

