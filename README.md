# 基于人脸检测与头部姿态识别的驾驶疲劳检测系统

项目使用Retinaface开源模型进行人脸检测与box获取；将box作为输入，使用6D头部姿态检测获取头部欧拉角信息。  
通过欧拉角度信息判断是否长时间低头，并发出警报避免疲劳驾驶。  
本项目基于pyqt构建显示界面及信息显示  

## 环境安装
可通过项目提供的requirements.txt安装所需环境。
```
pip install -r requirements.txt
```

## 权重获取
本项目提供Retinaface模型和sixdrepnet模型的预训练权重。获取方式如下：  
百度云链接: <https://pan.baidu.com/s/1w-5uUWd6UgGNvyc5HIt6bA?pwd=f3jn> 提取码: f3jn  
Google cloud：<https://drive.google.com/drive/folders/1AMRzcqO2D_igDYJq0qlHioDjs-l8p6wg?usp=drive_link>  

## 项目架构说明
HPEqt目录为pyqt项目文件  
Retinaface目录为人脸检测项目文件  
sixdrepnet目录为头部姿态识别项目文件  
weight目录存放模型权重文件  
resources目录存放项目资源文件

## 使用方法
直接运行app.py文件即可启动pyqt界面
```
python app.py
```

## 贡献

PRs accepted.

## 许可证

MIT © Richard McRichface
