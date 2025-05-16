---
title: 在Linux服务器上进行文件传输
date: 2025-05-16 10:00:00
tags:
  - 炼丹小技巧
  - 文件传输
categories:
  - 炼丹
---

# 在Linux服务器上进行文件传输

## 百度网盘

### 前置工作

bypy中转默认储存在百度云盘-我的应用数据-bypy文件夹里面

```
pip install bypy
bypy info
```

### 下载

```
//查看bypy文件夹里的文件
bypy list

//下载特定文件
bypy downfile filename

//下载文件夹下所有文件
bypy downdir -v
```

### 上传

```
bypy upload [localpath] [remotepath] [ondup] - upload a file or directory (recursively)
```

