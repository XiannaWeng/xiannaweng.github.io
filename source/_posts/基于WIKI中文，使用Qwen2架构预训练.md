---
title: 3090 24G显卡基于WIKI中文，使用Qwen2架构预训练
date: 2025-05-16 10:00:00
tags:
  - 预训练
categories:
  - 炼丹
---
# 3090 24G显卡基于WIKI中文，使用Qwen2架构预训练

第一次预训练，有很多代码细节不太理解，所以对代码做了全注释。

使用Hugging Face和魔搭的教程参考：
[魔搭社区和Hugging Face使用攻略](/2025/05/16/魔搭社区和Hugging-Face使用攻略)

```python
import datasets        # 导入Hugging Face datasets库，用于加载和处理数据集
import transformers   # 导入Hugging Face transformers库，用于加载模型、tokenizer和训练
import swanlab        # 导入swanlab库，用于实验跟踪和可视化
from swanlab.integration.huggingface import SwanLabCallback  # 导入SwanLab的Hugging Face回调函数
import modelscope     # 导入modelscope库，用于从魔搭平台加载模型和tokenizer
import os             # 导入os库，用于设置环境变量和文件操作

# 设置 Hugging Face 镜像地址为国内可访问的镜像站点
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def main():
    # 初始化SwanLab，用于跟踪和记录训练过程，"WikiLLM"是项目名称
    swanlab.init("WikiLLM")

    # 从本地JSON文件加载维基百科数据集
    # 这个文件包含中文维基百科的文章内容
    raw_datasets = datasets.load_dataset(
        "json", data_files="wikipedia-zh-cn-20240820.json"
    )

    # 将数据集分割为训练集和测试集
    # test_size=0.1表示10%的数据分配给测试集，90%用于训练
    # seed=2333确保每次运行时得到相同的随机分割结果
    raw_datasets = raw_datasets["train"].train_test_split(test_size=0.1, seed=2333)
    print("dataset info")  # 打印分割后的数据集信息
    print(raw_datasets)

    # 使用modelscope从魔搭平台下载Qwen2-0.5B模型的配置文件
    # 由于国内无法直接访问HuggingFace，因此使用modelscope作为替代
    modelscope.AutoConfig.from_pretrained("Qwen/Qwen2-0.5B").save_pretrained(
        "Qwen2-0.5B"  # 保存到本地目录
    )
    
    # 同样使用modelscope下载Qwen2-0.5B的tokenizer
    modelscope.AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B").save_pretrained(
        "Qwen2-0.5B"  # 保存到本地目录
    )
    
    # 设置上下文长度为384个token，这决定了模型能处理的文本长度
    context_length = 384  # 增加上下文长度以捕获更多依赖关系
    
    # 从本地加载刚下载的tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "./Qwen2-0.5B"
    )

    # 打印数据集中的第一个样本，以便了解数据结构
    print("First example from dataset:")
    print(raw_datasets["train"][0])

    # 定义tokenize函数，用于将文本转换为模型输入的数字形式
    def tokenize(element):
        # 使用tokenizer处理文本
        # truncation=True：如果文本超过最大长度，自动截断
        # max_length=context_length：设置最大长度
        # return_overflowing_tokens=True：如果文本超长，分割成多个片段
        # return_length=True：返回每个处理后片段的长度
        outputs = tokenizer(
            element["text"],  # 处理数据集中的"text"字段
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        
        # 创建一个空列表，用于存储符合长度要求的token序列
        input_batch = []
        
        # 遍历tokenizer处理后的所有文本片段
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            # 只保留长度恰好等于context_length的片段
            # 这确保了所有训练样本长度一致，有利于批处理效率
            if length == context_length:
                input_batch.append(input_ids)
                
        # 返回一个只包含input_ids的字典
        return {"input_ids": input_batch}

    # 使用map函数对整个数据集应用tokenize函数
    # batched=True：批量处理，提高效率
    # remove_columns：处理后删除原始列，只保留处理后的结果
    tokenized_datasets = raw_datasets.map(
        tokenize, batched=True, remove_columns=['id', 'title', 'tags', 'text']
    )
    print("tokenize dataset info")  # 打印tokenize后的数据集信息
    print(tokenized_datasets)
    
    # 设置padding token为end-of-sequence token
    # 这是因为Qwen2模型没有专门的padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    # 创建数据整理器，用于训练时的批处理
    # mlm=False表示使用因果语言模型（单向）而非掩码语言模型（双向）
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # 为从头训练模型准备配置
    # 从本地加载基础配置，但修改关键参数以适应我们的需求
    config = transformers.AutoConfig.from_pretrained(
        "./Qwen2-0.5B",
        vocab_size=len(tokenizer),  # 设置词汇表大小为tokenizer的词汇量
        hidden_size=768,         # 隐藏层维度，决定模型的表达能力
        intermediate_size=2048,  # 前馈网络中间层大小，影响模型复杂度
        num_attention_heads=12,  # 注意力头数量，影响模型捕获不同特征的能力
        num_hidden_layers=8,     # Transformer层数，决定模型深度
        n_ctx=context_length,    # 设置上下文长度
        bos_token_id=tokenizer.bos_token_id,  # 设置开始标记ID
        eos_token_id=tokenizer.eos_token_id,  # 设置结束标记ID
    )
    
    # 使用配置创建一个新的Qwen2因果语言模型
    model = transformers.Qwen2ForCausalLM(config)
    
    # 计算模型参数量并打印
    model_size = sum(t.numel() for t in model.parameters())
    print("Model Config:")
    print(config)
    print(f"Model Size: {model_size/1000**2:.1f}M parameters")

    # 设置训练参数
    args = transformers.TrainingArguments(
        # 模型保存的目录，训练过程中的检查点和最终模型会保存在这里
        output_dir="WikiLLM",
        
        # 每个GPU在训练时一次处理的样本数，较小的值可减少显存使用
        per_device_train_batch_size=4,
        
        # 每个GPU在评估时一次处理的样本数
        per_device_eval_batch_size=4,
        
        # 评估策略，设为"steps"表示每隔一定步数进行一次评估
        eval_strategy="steps",
        
        # 每训练500步进行一次评估
        eval_steps=5_00,
        
        # 每50步记录一次训练日志（损失等指标）
        logging_steps=50,
        
        # 梯度累积步数，实际的batch_size = per_device_batch_size * gradient_accumulation_steps
        # 这里设为8，相当于模拟了batch_size=32的训练效果，但只使用了batch_size=4的显存
        gradient_accumulation_steps=8,
        
        # 训练的总轮数，整个数据集将被完整训练3遍
        num_train_epochs=3,
        
        # 权重衰减率，用于L2正则化，防止过拟合
        weight_decay=0.05,
        
        # 学习率预热步数，初始阶段逐渐增加学习率，提高训练稳定性
        warmup_steps=500,
        
        # 使用的优化器，adamw是Adam优化器的一个变种，加入了权重衰减
        optim="adamw_torch", 
        
        # 学习率调度策略，"linear"表示学习率在预热后线性衰减到0
        lr_scheduler_type="linear",
        
        # 基础学习率，模型参数更新的步长
        learning_rate=3e-4,
        
        # 每训练500步保存一次模型检查点
        save_steps=5_00,
        
        # 最多保存5个检查点，超过后会删除最旧的，节省磁盘空间
        save_total_limit=5,
        
        # 启用bf16（brain float 16）混合精度训练，减少显存使用并加速训练
        bf16=True,
        
        # 启用梯度检查点技术，以牺牲一些计算速度为代价大幅减少显存使用
        # 在前向传播时不保存所有激活值，反向传播时重新计算需要的激活值
        gradient_checkpointing=True,
    )
    print("Train Args:")  # 打印训练参数
    print(args)
    
    # 创建Trainer实例，负责整个训练流程
    trainer = transformers.Trainer(
        model=model,                         # 要训练的模型
        tokenizer=tokenizer,                 # 使用的tokenizer
        args=args,                           # 训练参数
        data_collator=data_collator,         # 数据整理器，用于批处理
        train_dataset=tokenized_datasets["train"],  # 训练数据集
        eval_dataset=tokenized_datasets["test"],    # 评估数据集
        callbacks=[SwanLabCallback()],       # 添加SwanLab回调，用于可视化训练过程
    )
    
    # 启动训练过程
    trainer.train()

    # 将训练好的模型保存到指定路径
    model.save_pretrained("./WikiLLM/Weight")

    # 创建文本生成管道，用于测试模型生成能力
    # text-generation：指定任务类型为文本生成
    # model：使用刚训练好的模型
    # tokenizer：使用匹配的tokenizer
    # max_length=200：设置生成文本的最大长度
    pipe = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=200
    )
    
    # 使用"人工智能"作为提示词生成文本并打印
    # num_return_sequences=1：只生成一个结果
    # [0]：获取第一个生成结果
    # ["generated_text"]：获取生成的文本内容
    print("GENERATE:", pipe("人工智能", num_return_sequences=1)[0]["generated_text"])
    
    # 定义三个不同的提示词用于生成示例
    prompts = ["牛顿", "北京市", "亚洲历史"]
    
    # 创建空列表，用于收集生成的示例
    examples = []
    
    # 对每个提示词生成文本并添加到示例列表中
    for i in range(3):
        # 生成文本
        text = pipe(prompts[i], num_return_sequences=1)[0]["generated_text"]
        # 将文本转换为SwanLab的Text对象，便于在SwanLab中展示
        text = swanlab.Text(text)
        # 将文本对象添加到示例列表
        examples.append(text)
    
    # 使用SwanLab记录生成的文本示例，以便在SwanLab界面查看
    swanlab.log({"Generate": examples})


# 脚本入口点：当直接运行此脚本时执行main函数
if __name__ == "__main__":
    main()

```

