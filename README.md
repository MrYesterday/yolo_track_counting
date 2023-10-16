# yolo_track_counting
This project uses code from the [mikel-brostrom's Open Source Project:yolo_tracking](https://github.com/mikel-brostrom/yolo_tracking).
Special thanks to the contributors and maintainers for their valuable work.
结合摄像头和机器视觉技术，使用了 YOLO  目标检测深度模型来识别鱼类，并借助多目标跟踪算法来追踪这些目标。捕获到计数信息将自动发送控制信号给嵌入式单片机，进行必要的干预和控制。  
配置依赖：  
 ```
pip install -r requirements.txt  # install dependencies
```
仿照上述切换到对应目录安装所有依赖即可。  
如果使用与单片机通信的部分代码，安装与Raspberry通信的python依赖paramiko包：
https://pypi.org/project/paramiko/
如果出差错，可以查看下方博客指导手动安装：
python36之paramiko模块安装 - 流年似水zlw - 博客园 [(cnblogs.com)](https://www.cnblogs.com/zlw-xyz/p/12889212.html)https://www.cnblogs.com/zlw-xyz/p/12889212.html  
**可以使用拉流地址或者本地视频文件。  
输入测试视频：video/inputVid
输出结果：video/outputVid
碰线检测实现目标追踪的计数效果**
