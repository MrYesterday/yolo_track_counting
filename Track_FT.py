import argparse
import cv2
import os

import paramiko

# SSH连接参数
hostname = '192.168.2.1'
port = 22
username = 'username'
password = '8888'

# 本地文件路径
black_local_path = 'black_file.txt'
red_local_path = 'red_file.txt'
none_local_path = 'none_file.txt'

# 目标路径（树莓派上的路径）
black_remote_path = '/home/liwenjia/Desktop/black_file.txt'
red_remote_path = '/home/liwenjia/Desktop/red_file.txt'
none_remote_path = '/home/liwenjia/Desktop/none_file.txt'
# 创建SSH客户端
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())



# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov8') not in sys.path:
    sys.path.append(str(ROOT / 'yolov8'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov8.ultralytics.nn.autobackend import AutoBackend
from yolov8.ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadStreams
from yolov8.ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from yolov8.ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from yolov8.ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow, print_args, check_requirements
from yolov8.ultralytics.yolo.utils.files import increment_path
from yolov8.ultralytics.yolo.utils.torch_utils import select_device
from yolov8.ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes, process_mask, \
    process_mask_native
from yolov8.ultralytics.yolo.utils.plotting import Annotator, colors

from trackers.multi_tracker_zoo import create_tracker

import datetime
import threading

# 共享的变量，用于控制两个run函数的执行顺序
# lock = threading.Condition()

lock = threading.Lock()
cond = threading.Condition(lock)

front_red = 0
front_black = 0
flag = False  # 初始状态下第一个run函数先执行
running1 = True
running2 = True
choose_source = 0 # 选择视频源


Scaling = False
Zoom = 2



@torch.no_grad()
def run(  # 所有输入参数
        source='0',
        source2='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        tracking_method='strongsort',
        tracking_config=None,
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_trajectories=False,  # save trajectories for each track
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs' / 'track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride 帧速率步长，默认逐帧
        retina_masks=False,
):
    # 获取当前线程
    thread = threading.current_thread().ident
    global choose_source, running2
    if choose_source == 0:
        source = str(source)  # 读取数据源位置为字符串
    else:
        source = str(source2)  # 读取数据源位置为字符串
    save_img = not nosave and not source.endswith('.txt')  # save inference images 允许保存文件、源文件不是txt文件
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:  # 如果是一个url / 一个包含链接txt文件 下载这个视频
        source = check_file(source)  # download

    # 打开视频文件 获取帧速率
    fps = cv2.VideoCapture(source).get(cv2.CAP_PROP_FPS)

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model 不是list
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + reid_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # ----------------分割线--------------
    imgsz = (int(cv2.VideoCapture(source).get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(cv2.VideoCapture(source).get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if Scaling:
        imgsz[0] = imgsz[0]//Zoom
        imgsz[1] = imgsz[1]//Zoom
    left = 0
    right = 1
    mask_image_temp = np.zeros((imgsz[1], imgsz[0]), dtype=np.uint8)
    # list_pts_blue = [[720, 0], [680, 0], [680, 1080], [720, 1080]]
    list_pts_blue = [[imgsz[0] // 2 * right, 0], [imgsz[0] // 2 * left, 0], [imgsz[0] // 2 * left, imgsz[1]],
                     [imgsz[0] // 2 * right, imgsz[1]]]
    ndarray_pts_blue = np.array(list_pts_blue, np.int32)
    polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)
    polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]

    # 填充第二个polygon
    left = 1
    right = 2
    mask_image_temp = np.zeros((imgsz[1], imgsz[0]), dtype=np.uint8)
    list_pts_yellow = [[imgsz[0] // 2 * right, 0], [imgsz[0] // 2 * left, 0], [imgsz[0] // 2 * left, imgsz[1]],
                       [imgsz[0] // 2 * right, imgsz[1]]]
    # list_pts_yellow = [[181, 305], [207, 442], [603, 544], [1107, 485], [1898, 625], [1893, 701], [1101, 568],
    #                    [594, 637], [118, 483], [109, 303]]
    ndarray_pts_yellow = np.array(list_pts_yellow, np.int32)
    polygon_yellow_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color=2)
    polygon_yellow_value_2 = polygon_yellow_value_2[:, :, np.newaxis]
    # 撞线检测用mask，包含2个polygon，（值范围 0、1、2），供撞线计算使用
    polygon_mask_blue_and_yellow = polygon_blue_value_1 + polygon_yellow_value_2




    # 设置图像中显示的碰撞线 （窄区）
    left = 0.95
    right = 1
    mask_image_temp = np.zeros((imgsz[1], imgsz[0]), dtype=np.uint8)
    # list_pts_blue = [[720, 0], [680, 0], [680, 1080], [720, 1080]]
    list_pts_blue = [[imgsz[0] // 2 * right, 0], [imgsz[0] // 2 * left, 0], [imgsz[0] // 2 * left, imgsz[1]],
                     [imgsz[0] // 2 * right, imgsz[1]]]
    ndarray_pts_blue = np.array(list_pts_blue, np.int32)
    polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)
    polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]

    # 填充第二个polygon
    left = 1
    right = 1.05
    mask_image_temp = np.zeros((imgsz[1], imgsz[0]), dtype=np.uint8)
    list_pts_yellow = [[imgsz[0] // 2 * right, 0], [imgsz[0] // 2 * left, 0], [imgsz[0] // 2 * left, imgsz[1]],
                       [imgsz[0] // 2 * right, imgsz[1]]]
    # list_pts_yellow = [[181, 305], [207, 442], [603, 544], [1107, 485], [1898, 625], [1893, 701], [1101, 568],
    #                    [594, 637], [118, 483], [109, 303]]
    ndarray_pts_yellow = np.array(list_pts_yellow, np.int32)
    polygon_yellow_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color=2)
    polygon_yellow_value_2 = polygon_yellow_value_2[:, :, np.newaxis]


    # 蓝 色盘 b,g,r
    blue_color_plate = [255, 0, 0]
    # 蓝 polygon图片
    blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)

    # 黄 色盘
    yellow_color_plate = [0, 255, 255]
    # 黄 polygon图片
    yellow_image = np.array(polygon_yellow_value_2 * yellow_color_plate, np.uint8)

    # 彩色图片（值范围 0-255）
    color_polygons_image = blue_image + yellow_image
    # 缩小尺寸
    color_polygons_image = cv2.resize(color_polygons_image, imgsz)

    # list 与蓝色polygon重叠
    list_overlapping_blue_polygon = []

    # list 与黄色polygon重叠
    list_overlapping_yellow_polygon = []

    # 两类鱼计数数量
    down_count_red = 0
    down_count_black = 0


    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX

    # ----------------------------分割线--------------------------------

    # Load model
    device = select_device(device)
    model = AutoBackend(yolo_weights, device=device, dnn=dnn, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_imgsz(imgsz, stride=stride)  # check image size
    print(model.names)
    # Dataloader
    bs = 1
    if webcam:
        show_vid = check_imshow(warn=True)
        dataset = LoadStreams(
            source,
            imgsz=imgsz,
            stride=stride,
            auto=pt,
            transforms=getattr(model.model, 'transforms', None),
            vid_stride=vid_stride
        )
        bs = len(dataset)
    else:
        dataset = LoadImages(
            source,
            imgsz=imgsz,
            stride=stride,
            auto=pt,
            transforms=getattr(model.model, 'transforms', None),
            vid_stride=vid_stride
        )
    vid_path, vid_writer, txt_path = [None] * bs, [None] * bs, [None] * bs
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup

    # Create as many strong sort instances as there are video sources
    tracker_list = []
    for i in range(bs):
        tracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)
        # strongSort None 超参数 产生与数据集个数匹配的追踪器列表
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()
    outputs = [None] * bs

    # Run tracking
    # model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
    curr_frames, prev_frames = [None] * bs, [None] * bs

    unitTime = 30

    queue_black = [0 for i in range(int(unitTime*fps))]
    queue_red = [0 for i in range(int(unitTime*fps))]
    RecordedTarget = []  # 已计数目标
    for frame_idx, batch in enumerate(dataset):  # 启动追踪器处理数据集每一帧，存储到p变量中
        black_change = False
        red_change = False
        path, im, im0s, vid_cap, s = batch
        # 是否可视化特征图
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        # 视频图像整型数据转浮点
        with dt[0]:
            im = torch.from_numpy(im).to(device)  # 根据设备调整图像设备位置
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0 转浮点
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
        if Scaling:
            im = cv2.resize(im,imgsz)
        # 模型接口调用，输入视频图像数据，获得对应视频张量输出
        # Inference
        with dt[1]:
            preds = model(im, augment=augment, visualize=visualize)

        # Apply NMS 非极大值抑制，保留极大值元素
        with dt[2]:
            # 对输出的张量进行非极大值抑制
            p = non_max_suppression(preds, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process detections 数据集探测结果逐帧处理
        for i, det in enumerate(p):  # detections per image 对所有路径进行处理
            seen += 1

            if webcam:  # bs >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS): # 源是视频文件
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else: # 其他类型文件 获取其父文件夹
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...


            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            # 新建一个解释器
            if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
                if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation 不是第一帧就进行补偿
                    tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):  # 如果画面中有bbox，检测不为空，处理所有检测框
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # pass detections to strongsort 把预测结果交给strongsort
                # 如果画面中有bbox
                with dt[3]:  # 更新当前追踪物体列表
                    outputs[i] = tracker_list[i].update(det.cpu(), im0)

                # draw boxes for visualization 可视化矩形框
                if len(outputs[i]) > 0:

                    tmpOut = outputs[i]
                    trackIDCurrent = [] # 获取当前帧所有目标的ID
                    for j, output in enumerate(tmpOut):
                        if output[4] not in trackIDCurrent:
                            trackIDCurrent.append(output[4])
                    for j, output in enumerate(tmpOut):  # 遍历所有检测出的物体

                        # -------------分割线------------------
                        x1 = output[0]
                        y1 = output[1]
                        x2 = output[2]
                        y2 = output[3]
                        track_id = output[4]  # 追踪物体id
                        # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                        y_offset = y1 + ((y2 - y1) * 0.5)
                        x_offset = x1 + ((x2 - x1) * 0.5)
                        # 撞线的点
                        y = int(y_offset)
                        x = int(x_offset)
                        # if Scaling:
                        #     y = y//Zoom
                        #     x = x//Zoom
                        # y = int((y1+y2)/2)
                        # x = int((x1+x2)/2)
                        if polygon_mask_blue_and_yellow[y, x] == 1:
                            # 如果撞 蓝polygon
                            if track_id not in list_overlapping_blue_polygon:
                                list_overlapping_blue_polygon.append(track_id)
                            pass
                        elif polygon_mask_blue_and_yellow[y, x] == 2:
                            # 如果撞 黄polygon
                            if track_id not in list_overlapping_yellow_polygon:
                                list_overlapping_yellow_polygon.append(track_id)
                            pass

                            # 判断 蓝polygon list 里是否有此 track_id
                            # 有此 track_id，则 认为是 进入方向
                            if track_id in list_overlapping_blue_polygon:
                                if track_id in RecordedTarget:
                                    break
                                else:
                                    RecordedTarget.append(track_id)
                                # 进入+1
                                if output[5] == 1:  # black koi
                                    down_count_black += 1
                                    black_change = True
                                    print('down count black:', down_count_black, ', down id:', list_overlapping_blue_polygon)
                                if output[5] == 0:  # red koi
                                    down_count_red += 1
                                    red_change = True
                                    print('down count red:', down_count_red, ', down id:', list_overlapping_blue_polygon)

                                # 删除 蓝polygon list 中的此id
                                list_overlapping_blue_polygon.remove(track_id)
                                pass
                            else:
                                # 无此 track_id，不做其他操作
                                pass
                            pass
                        else:
                            pass
                        pass

                        # ---------------------分割线-------------------------

                        bbox = output[0:4]  # YOLO矩形框参数
                        id = output[4]  # 追踪物体id
                        cls = output[5]
                        conf = output[6]

                        if save_txt:
                            # to MOT format YOLO矩形框各个参数
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                        if save_vid or show_vid:  # Add bbox/seg to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                                                  (
                                                                      f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                            color = colors(c, True)  # 设置标签
                            annotator.box_label(bbox, label, color=color)  # 解释器对该帧进行bbox标注

                            if save_trajectories and tracking_method == 'strongsort':
                                q = output[7]
                                tracker_list[i].trajectory(im0, q, color=color)

                    # ----------------------清除无用id----------------------
                    list_overlapping_all = list_overlapping_yellow_polygon + list_overlapping_blue_polygon
                    for id1 in list_overlapping_all:
                        is_found = False
                        for j, output in enumerate(outputs[i]):
                            track_id = output[4]
                            if track_id == id1:
                                is_found = True
                                break
                            pass
                        pass

                        if not is_found:
                            # 如果没找到，删除id
                            if id1 in list_overlapping_yellow_polygon:
                                list_overlapping_yellow_polygon.remove(id1)
                            pass
                            if id1 in list_overlapping_blue_polygon:
                                list_overlapping_blue_polygon.remove(id1)
                            pass
                        pass
                    list_overlapping_all.clear()
                    for k in RecordedTarget:# 遍历当前帧是否有已计数的目标，如果没有删除这个已被记录的目标id
                        if k not in trackIDCurrent:
                            RecordedTarget.remove(k)
                    # ---------------------分割线------------------------
            else:
                # 如果图像中没有任何的bbox，则清空list
                list_overlapping_blue_polygon.clear()
                list_overlapping_yellow_polygon.clear()
                pass

            # Stream results
            im0 = annotator.result()
            # 将当前帧的统计信息加入队列末尾
            queue_red.append(down_count_red)
            queue_black.append(down_count_black)
            # 如果队列长度超过n，则将队列头部的元素删除
            if len(queue_red) > (unitTime*fps + 1):
                queue_red.pop(0)
            if len(queue_black) > (unitTime*fps + 1):
                queue_black.pop(0)
            down_throughput_red = (down_count_red - queue_red[0])
            down_throughput_black = (down_count_black - queue_black[0])
            global front_red, front_black
            front_red = down_throughput_red
            front_black = down_throughput_black

            if black_change:
                # ssh发送文件
                try:
                    with open(black_local_path, 'w') as f:
                        f.writelines(str(down_throughput_black))

                    # 连接SSH服务器
                    client.connect(hostname, port, username, password)

                    # 创建SFTP会话
                    sftp = client.open_sftp()

                    # 上传文件
                    sftp.put(black_local_path, black_remote_path)

                    # 关闭SFTP会话和SSH连接
                    sftp.close()
                    client.close()

                    print('文件传输成功！')

                except paramiko.AuthenticationException:
                    print('认证失败，请检查用户名和密码。')
                except paramiko.SSHException as ssh_exception:
                    print('SSH连接错误:', str(ssh_exception))
                # except paramiko.SFTPException as sftp_exception:
                #     print('SFTP操作错误:', str(sftp_exception))
                except Exception as e:
                    print('错误:', str(e))

            if red_change:
                # ssh发送文件
                try:
                    with open(red_local_path, 'w') as f:
                        f.writelines(str(down_throughput_red))

                    # 连接SSH服务器
                    client.connect(hostname, port, username, password)

                    # 创建SFTP会话
                    sftp = client.open_sftp()

                    # 上传文件
                    sftp.put(red_local_path, red_remote_path)

                    # 关闭SFTP会话和SSH连接
                    sftp.close()
                    client.close()

                    print('文件传输成功！')

                except paramiko.AuthenticationException:
                    print('认证失败，请检查用户名和密码。')
                except paramiko.SSHException as ssh_exception:
                    print('SSH连接错误:', str(ssh_exception))
                # except paramiko.SFTPException as sftp_exception:
                #     print('SFTP操作错误:', str(sftp_exception))
                except Exception as e:
                    print('错误:', str(e))

            if queue_black[-1]==queue_black[0] and queue_red[-1]==queue_red[0]: # 无鱼
                # ssh发送文件
                try:
                    with open(none_local_path, 'w') as f:
                        f.writelines(str(0))

                    # 连接SSH服务器
                    client.connect(hostname, port, username, password)

                    # 创建SFTP会话
                    sftp = client.open_sftp()

                    # 上传文件
                    sftp.put(none_local_path, none_remote_path)

                    # 关闭SFTP会话和SSH连接
                    sftp.close()
                    client.close()

                    print('文件传输成功！')

                except paramiko.AuthenticationException:
                    print('认证失败，请检查用户名和密码。')
                except paramiko.SSHException as ssh_exception:
                    print('SSH连接错误:', str(ssh_exception))
                # except paramiko.SFTPException as sftp_exception:
                #     print('SFTP操作错误:', str(sftp_exception))
                except Exception as e:
                    print('错误:', str(e))


            if show_vid:
                # --------------------分割线------------------------
                text_draw_red = 'Fish A: ' + str(round(down_throughput_red, 2))
                draw_text_postion = (int(im0.shape[1] * 0.01), int(im0.shape[0] * 0.1))
                im0 = cv2.putText(img=im0, text=text_draw_red,
                                  org=draw_text_postion,
                                  fontFace=font_draw_number,
                                  fontScale=3, color=(255, 255, 255), thickness=7)
                im0 = cv2.add(im0, cv2.resize(cv2.resize(color_polygons_image, imgsz), (im0.shape[1], im0.shape[0])))
                # --------------------分割线------------------------
                text_draw_black = 'Fish B: ' + str(round(down_throughput_black, 2))
                draw_text_postion = (int(im0.shape[1] * 0.55), int(im0.shape[0] * 0.1))
                im0 = cv2.putText(img=im0, text=text_draw_black,
                                  org=draw_text_postion,
                                  fontFace=font_draw_number,
                                  fontScale=3, color=(255, 255, 255), thickness=7)
                im0 = cv2.add(im0, cv2.resize(cv2.resize(color_polygons_image, imgsz), (im0.shape[1], im0.shape[0])))
                # --------------------分割线------------------------
                # --------------------分割线------------------------
                # print("Current thread ID:", threading.current_thread()g.ident)
                if Scaling:
                    imgsz[0] = imgsz[0] // Zoom
                    imgsz[1] = imgsz[1] // Zoom
                cv2.imshow("Thread 1 from "+str(p), cv2.resize(im0, imgsz))

                if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                    exit()

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]  # 当前帧作为前一帧

        # Print total time (preprocessing + inference + NMS + tracking)
        LOGGER.info(
            f"{s}{'' if len(det) else '(no detections), '}{sum([dt.dt for dt in dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms")

        # # 在每一帧处理完之后修改 flag，唤醒另一个线程
        # # print("Thread 1 is running")
        # if running2:
        #     lock.acquire()
        #     global flag
        #     while not flag:
        #         cond.wait()
        #     flag = not flag
        #     cond.notify()  # 唤醒另一个线程
        #     lock.release()




    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image 帧速率
    LOGGER.info(
        f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {tracking_method} update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list((save_dir / 'tracks').glob('*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    # global running1
    # running1 = False



@torch.no_grad()
def run2(  # 所有输入参数
        source='0',
        source2='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        tracking_method='strongsort',
        tracking_config=None,
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_trajectories=False,  # save trajectories for each track
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs' / 'track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride 帧速率步长，默认逐帧
        retina_masks=False,
):
    global choose_source, running1
    if choose_source == 0:
        source = str(source)  # 读取数据源位置为字符串
    else:
        source = str(source2)  # 读取数据源位置为字符串
    save_img = not nosave and not source.endswith('.txt')  # save inference images 允许保存文件、源文件不是txt文件
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:  # 如果是一个url / 一个包含链接txt文件 下载这个视频
        source = check_file(source)  # download

    # 打开视频文件 获取帧速率
    fps = cv2.VideoCapture(source).get(cv2.CAP_PROP_FPS)

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model 不是list
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + reid_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # ----------------分割线--------------
    imgsz = (int(cv2.VideoCapture(source).get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(cv2.VideoCapture(source).get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if Scaling:
        imgsz[0] = imgsz[0]//Zoom
        imgsz[1] = imgsz[1]//Zoom
    left = 0
    right = 1
    mask_image_temp = np.zeros((imgsz[1], imgsz[0]), dtype=np.uint8)
    # list_pts_blue = [[720, 0], [680, 0], [680, 1080], [720, 1080]]
    list_pts_blue = [[imgsz[0] // 2 * right, 0], [imgsz[0] // 2 * left, 0], [imgsz[0] // 2 * left, imgsz[1]],
                     [imgsz[0] // 2 * right, imgsz[1]]]
    ndarray_pts_blue = np.array(list_pts_blue, np.int32)
    polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)
    polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]

    # 填充第二个polygon
    left = 1
    right = 2
    mask_image_temp = np.zeros((imgsz[1], imgsz[0]), dtype=np.uint8)
    list_pts_yellow = [[imgsz[0] // 2 * right, 0], [imgsz[0] // 2 * left, 0], [imgsz[0] // 2 * left, imgsz[1]],
                       [imgsz[0] // 2 * right, imgsz[1]]]
    # list_pts_yellow = [[181, 305], [207, 442], [603, 544], [1107, 485], [1898, 625], [1893, 701], [1101, 568],
    #                    [594, 637], [118, 483], [109, 303]]
    ndarray_pts_yellow = np.array(list_pts_yellow, np.int32)
    polygon_yellow_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color=2)
    polygon_yellow_value_2 = polygon_yellow_value_2[:, :, np.newaxis]

    # 撞线检测用mask，包含2个polygon，（值范围 0、1、2），供撞线计算使用
    polygon_mask_blue_and_yellow = polygon_blue_value_1 + polygon_yellow_value_2

    # 设置图像中显示的碰撞线 （窄区）
    left = 0.90
    right = 0.95
    mask_image_temp = np.zeros((imgsz[1], imgsz[0]), dtype=np.uint8)
    # list_pts_blue = [[720, 0], [680, 0], [680, 1080], [720, 1080]]
    list_pts_blue = [[imgsz[0] // 2 * right, 0], [imgsz[0] // 2 * left, 0], [imgsz[0] // 2 * left, imgsz[1]],
                     [imgsz[0] // 2 * right, imgsz[1]]]
    ndarray_pts_blue = np.array(list_pts_blue, np.int32)
    polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)
    polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]

    # 填充第二个polygon
    left = 0.95
    right = 1.00
    mask_image_temp = np.zeros((imgsz[1], imgsz[0]), dtype=np.uint8)
    list_pts_yellow = [[imgsz[0] // 2 * right, 0], [imgsz[0] // 2 * left, 0], [imgsz[0] // 2 * left, imgsz[1]],
                       [imgsz[0] // 2 * right, imgsz[1]]]
    # list_pts_yellow = [[181, 305], [207, 442], [603, 544], [1107, 485], [1898, 625], [1893, 701], [1101, 568],
    #                    [594, 637], [118, 483], [109, 303]]
    ndarray_pts_yellow = np.array(list_pts_yellow, np.int32)
    polygon_yellow_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color=2)
    polygon_yellow_value_2 = polygon_yellow_value_2[:, :, np.newaxis]

    # 蓝 色盘 b,g,r
    blue_color_plate = [255, 0, 0]
    # 蓝 polygon图片
    blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)

    # 黄 色盘
    yellow_color_plate = [0, 255, 255]
    # 黄 polygon图片
    yellow_image = np.array(polygon_yellow_value_2 * yellow_color_plate, np.uint8)

    # 彩色图片（值范围 0-255）
    color_polygons_image = blue_image + yellow_image
    # 缩小尺寸
    color_polygons_image = cv2.resize(color_polygons_image, imgsz)

    # list 与蓝色polygon重叠
    list_overlapping_blue_polygon = []

    # list 与黄色polygon重叠
    list_overlapping_yellow_polygon = []

    # 两类鱼计数数量
    down_count_red = 0
    down_count_black = 0


    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX

    # ----------------------------分割线--------------------------------

    # Load model
    device = select_device(device)
    model = AutoBackend(yolo_weights, device=device, dnn=dnn, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_imgsz(imgsz, stride=stride)  # check image size
    print(model.names)
    # Dataloader
    bs = 1
    if webcam:
        show_vid = check_imshow(warn=True)
        dataset = LoadStreams(
            source,
            imgsz=imgsz,
            stride=stride,
            auto=pt,
            transforms=getattr(model.model, 'transforms', None),
            vid_stride=vid_stride
        )
        bs = len(dataset)
    else:
        dataset = LoadImages(
            source,
            imgsz=imgsz,
            stride=stride,
            auto=pt,
            transforms=getattr(model.model, 'transforms', None),
            vid_stride=vid_stride
        )
    vid_path, vid_writer, txt_path = [None] * bs, [None] * bs, [None] * bs
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup

    # Create as many strong sort instances as there are video sources
    tracker_list = []
    for i in range(bs):
        tracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)
        # strongSort None 超参数 产生与数据集个数匹配的追踪器列表
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()
    outputs = [None] * bs

    # Run tracking
    # model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
    curr_frames, prev_frames = [None] * bs, [None] * bs

    unitTime = 20
    logInterval = 3
    # 创建log目录
    if not os.path.exists('log'):
        os.makedirs('log')

    # 获取当前日期和时间信息
    now = datetime.datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    countFrame = 0
    last_front_red = 0
    last_front_black = 0
    last_red = 0
    last_black = 0
    queue_black = [0 for i in range(int(unitTime*fps))]
    queue_red = [0 for i in range(int(unitTime*fps))]
    RecordedTarget = []  # 已计数目标
    for frame_idx, batch in enumerate(dataset):  # 启动追踪器处理数据集每一帧，存储到p变量中
        path, im, im0s, vid_cap, s = batch
        # 是否可视化特征图
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        # 视频图像整型数据转浮点
        with dt[0]:
            im = torch.from_numpy(im).to(device)  # 根据设备调整图像设备位置
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0 转浮点
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
        if Scaling:
            im = cv2.resize(im,imgsz)
        # 模型接口调用，输入视频图像数据，获得对应视频张量输出
        # Inference
        with dt[1]:
            preds = model(im, augment=augment, visualize=visualize)

        # Apply NMS 非极大值抑制，保留极大值元素
        with dt[2]:
            # 对输出的张量进行非极大值抑制
            p = non_max_suppression(preds, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process detections 数据集探测结果逐帧处理
        for i, det in enumerate(p):  # detections per image 对所有路径进行处理
            seen += 1

            if webcam:  # bs >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS): # 源是视频文件
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else: # 其他类型文件 获取其父文件夹
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...


            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            # 新建一个解释器
            if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
                if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation 不是第一帧就进行补偿
                    tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):  # 如果画面中有bbox，检测不为空，处理所有检测框
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # pass detections to strongsort 把预测结果交给strongsort
                # 如果画面中有bbox
                with dt[3]:  # 更新当前追踪物体列表
                    outputs[i] = tracker_list[i].update(det.cpu(), im0)

                # draw boxes for visualization 可视化矩形框
                if len(outputs[i]) > 0:

                    tmpOut = outputs[i]
                    trackIDCurrent = [] # 获取当前帧所有目标的ID
                    for j, output in enumerate(tmpOut):
                        if output[4] not in trackIDCurrent:
                            trackIDCurrent.append(output[4])
                    for j, output in enumerate(tmpOut):  # 遍历所有检测出的物体

                        # -------------分割线------------------
                        x1 = output[0]
                        y1 = output[1]
                        x2 = output[2]
                        y2 = output[3]
                        track_id = output[4]  # 追踪物体id
                        # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                        y_offset = y1 + ((y2 - y1) * 0.5)
                        x_offset = x1 + ((x2 - x1) * 0.5)
                        # 撞线的点
                        y = int(y_offset)
                        x = int(x_offset)
                        # if Scaling:
                        #     y = y//Zoom
                        #     x = x//Zoom
                        # y = int((y1+y2)/2)
                        # x = int((x1+x2)/2)
                        if polygon_mask_blue_and_yellow[y, x] == 1:
                            # 如果撞 蓝polygon
                            if track_id not in list_overlapping_blue_polygon:
                                list_overlapping_blue_polygon.append(track_id)
                            pass
                        elif polygon_mask_blue_and_yellow[y, x] == 2:
                            # 如果撞 黄polygon
                            if track_id not in list_overlapping_yellow_polygon:
                                list_overlapping_yellow_polygon.append(track_id)
                            pass

                            # 判断 蓝polygon list 里是否有此 track_id
                            # 有此 track_id，则 认为是 进入方向
                            if track_id in list_overlapping_blue_polygon:
                                if track_id in RecordedTarget:
                                    break
                                else:
                                    RecordedTarget.append(track_id)
                                # 进入+1
                                if output[5] == 1:  # black koi
                                    down_count_black += 1
                                    print('down count black:', down_count_black, ', down id:', list_overlapping_blue_polygon)
                                if output[5] == 0:  # red koi
                                    down_count_red += 1
                                    print('down count red:', down_count_red, ', down id:', list_overlapping_blue_polygon)
                                # 删除 蓝polygon list 中的此id
                                list_overlapping_blue_polygon.remove(track_id)
                                pass
                            else:
                                # 无此 track_id，不做其他操作
                                pass
                            pass
                        else:
                            pass
                        pass

                        # ---------------------分割线-------------------------

                        bbox = output[0:4]  # YOLO矩形框参数
                        id = output[4]  # 追踪物体id
                        cls = output[5]
                        conf = output[6]


                        if save_txt:
                            # to MOT format YOLO矩形框各个参数
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                        if save_vid or show_vid:  # Add bbox/seg to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                            (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                            color = colors(c, True)  # 设置标签
                            annotator.box_label(bbox, label, color=color)  # 解释器对该帧进行bbox标注

                            if save_trajectories and tracking_method == 'strongsort':
                                q = output[7]
                                tracker_list[i].trajectory(im0, q, color=color)

                    # ----------------------清除无用id----------------------
                    list_overlapping_all = list_overlapping_yellow_polygon + list_overlapping_blue_polygon
                    for id1 in list_overlapping_all:
                        is_found = False
                        for j, output in enumerate(outputs[i]):
                            track_id = output[4]
                            if track_id == id1:
                                is_found = True
                                break
                            pass
                        pass

                        if not is_found:
                            # 如果没找到，删除id
                            if id1 in list_overlapping_yellow_polygon:
                                list_overlapping_yellow_polygon.remove(id1)
                            pass
                            if id1 in list_overlapping_blue_polygon:
                                list_overlapping_blue_polygon.remove(id1)
                            pass
                        pass
                    list_overlapping_all.clear()
                    for k in RecordedTarget:# 遍历当前帧是否有已计数的目标，如果没有删除这个已被记录的目标id
                        if k not in trackIDCurrent:
                            RecordedTarget.remove(k)


                    # ---------------------分割线------------------------
            else:
                # 如果图像中没有任何的bbox，则清空list
                list_overlapping_blue_polygon.clear()
                list_overlapping_yellow_polygon.clear()
                pass

            # Stream results
            im0 = annotator.result()
            # 将当前帧的统计信息加入队列末尾q
            queue_red.append(down_count_red)
            queue_black.append(down_count_black)
            # 如果队列长度超过n，则将队列头部的元素删除
            if len(queue_red) > (unitTime * fps + 1):
                queue_red.pop(0)
            if len(queue_black) > (unitTime * fps + 1):
                queue_black.pop(0)
            down_throughput_red = (down_count_red - queue_red[0])
            down_throughput_black = (down_count_black - queue_black[0])
            global front_red, front_black
            if show_vid:
                # --------------------分割线------------------------
                text_draw_red = 'Fish A: ' + str(round(down_throughput_red, 2))
                draw_text_postion = (int(im0.shape[1] * 0.01), int(im0.shape[0] * 0.1))
                im0 = cv2.putText(img=im0, text=text_draw_red,
                                  org=draw_text_postion,
                                  fontFace=font_draw_number,
                                  fontScale=3, color=(255, 255, 255), thickness=7)
                im0 = cv2.add(im0, cv2.resize(cv2.resize(color_polygons_image, imgsz), (im0.shape[1], im0.shape[0])))
                # --------------------分割线------------------------
                text_draw_black = 'Fish B: ' + str(round(down_throughput_black, 2))
                draw_text_postion = (int(im0.shape[1] * 0.55), int(im0.shape[0] * 0.1))
                im0 = cv2.putText(img=im0, text=text_draw_black,
                                  org=draw_text_postion,
                                  fontFace=font_draw_number,
                                  fontScale=3, color=(255, 255, 255), thickness=7)
                im0 = cv2.add(im0, cv2.resize(cv2.resize(color_polygons_image, imgsz), (im0.shape[1], im0.shape[0])))
                # --------------------分割线------------------------
                # --------------------分割线------------------------
                text_draw_red = 'passingRate:' + \
                                (str(round(down_throughput_red / front_red, 2)) if front_red != 0 else '0')
                draw_text_postion = (int(im0.shape[1] * 0.01), int(im0.shape[0] * 0.2))
                im0 = cv2.putText(img=im0, text=text_draw_red,
                                  org=draw_text_postion,
                                  fontFace=font_draw_number,
                                  fontScale=2, color=(255, 255, 255), thickness=7)
                im0 = cv2.add(im0,
                              cv2.resize(cv2.resize(color_polygons_image, imgsz), (im0.shape[1], im0.shape[0])))
                # --------------------分割线------------------------
                text_draw_black = 'passingRate:' + \
                                  (str(round(down_throughput_black / front_black, 2)) if front_black != 0 else '0')
                draw_text_postion = (int(im0.shape[1] * 0.50), int(im0.shape[0] * 0.2))
                im0 = cv2.putText(img=im0, text=text_draw_black,
                                  org=draw_text_postion,
                                  fontFace=font_draw_number,
                                  fontScale=2, color=(255, 255, 255), thickness=7)
                im0 = cv2.add(im0,
                              cv2.resize(cv2.resize(color_polygons_image, imgsz), (im0.shape[1], im0.shape[0])))


                # --------------------分割线------------------------
                # print("Current thread ID:", threading.current_thread().ident)
                if Scaling:
                    imgsz[0] = imgsz[0] // Zoom
                    imgsz[1] = imgsz[1] // Zoom
                cv2.imshow("Thread 2 from "+str(p), cv2.resize(im0, imgsz))
                if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                    exit()

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]  # 当前帧作为前一帧

        # Print total time (preprocessing + inference + NMS + tracking)
        LOGGER.info(
            f"{s}{'' if len(det) else '(no detections), '}{sum([dt.dt for dt in dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms")
        countFrame = countFrame + 1
        countFrame = countFrame % (fps * logInterval)
        if countFrame == 0:
            # 将变量名和内容写入文件

            filename = f"log/log-{dt_string}.txt"
            with open(filename, 'a') as f:
                now = datetime.datetime.now()
                time = now.strftime("%H-%M-%S")
                f.write(f"{time} ")
                f.write(f"Fish A: {down_throughput_red}, "
                        f"总过鱼率: "
                        f"{(str(round(down_count_red / last_front_red, 2)) if last_front_red != 0 else '0')} "
                        f"单位时间过鱼率: "
                        f"{(str(round((down_count_red - last_red) / last_front_red, 2)) if last_front_red != 0 else '0')} ")
                f.write(f"Fish B: {down_throughput_black}, "
                        f"总过鱼率: "
                        f"{(str(round(down_count_black / last_front_black, 2)) if last_front_black != 0 else '0')} "
                        f"单位时间过鱼率: "
                        f"{(str(round((down_count_black - last_black) / last_front_black, 2)) if last_front_black != 0 else '0')} \n")

            last_front_red = front_red
            last_front_black = front_black
            last_red = down_count_red
            last_black = down_count_black
        # 在每一帧处理完之后修改 flag，唤醒另一个线程
        # print("Thread 2 is running")
        # if running1:
        #     lock.acquire()
        #     global flag
        #     while flag:
        #         cond.wait()
        #     flag = not flag
        #     cond.notify()  # 唤醒另一个线程
        #     lock.release()



    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image 帧速率
    LOGGER.info(
        f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {tracking_method} update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list((save_dir / 'tracks').glob('*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")

    # global running2
    # running2 = False

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--yolo-weights', type=Path, default='passrate-update.pt', help='model.pt path(s)')

    parser.add_argument('--reid-weights', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--tracking-method', type=str, default='bytetrack', help='strongsort, ocsort, bytetrack')
    parser.add_argument('--tracking-config', type=Path, default=None)


    # parser.add_argument('--source', type=str,
    #                     default='https://ali-adaptive.pull.yximgs.com/gifshow/kwai_actL_ol_act_11081135941_strL_hd2000.flv?auth_key=1685870812-0-0-4decb899cadab129df85ba057ac59526&tsc=origin&oidc=watchmen&sidc=204180&srcStrm=JfSpzxeNxUI&fd=1&ss=s19&kabr_spts=-5000'
    #                     ,help='file/dir/URL/glob, 0 for webcam')
    # parser.add_argument('--source2', type=str,
    #                     default='https://ali-adaptive.pull.yximgs.com/gifshow/kwai_actL_ol_act_11081552190_strL_hd2000.flv?auth_key=1685870860-0-0-151ea9d1b1f15c180a621fc8fc99f204&tsc=origin&oidc=watchmen&sidc=206077&srcStrm=uXcOKNSwMC0&fd=1&ss=s19&kabr_spts=-5000',
    #                     help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--source', type=str,
                        default='v1.mp4',help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--source2', type=str,
                        default='v2.mp4',help='file/dir/URL/glob, 0 for webcam')


    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[960, 544],
                        help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.3, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument('--show-vid', default=True, help='display tracking video results')
    # parser.add_argument('--show-vid', action='store_true', help='display tracking video results')

    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-trajectories', action='store_true', help='save trajectories for each track')

    parser.add_argument('--save-vid', default=True, help='save video tracking results')
    # parser.add_argument('--save-vid', action='store_true', help='save video tracking results')

    parser.add_argument('--nosave', default='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--retina-masks', action='store_true', help='whether to plot masks in native resolution')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    opt.tracking_config = ROOT / 'trackers' / opt.tracking_method / 'configs' / (opt.tracking_method + '.yaml')
    print_args(vars(opt))
    return opt

def run_with_opt(opt):
    run(**vars(opt))
def run_with_opt_2(opt):
    run2(**vars(opt))

def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    # 创建两个线程，分别运行run_with_opt()函数
    # global choose_source
    t1 = threading.Thread(target=run_with_opt, args=(opt,))
    # t2 = threading.Thread(target=run_with_opt_2, args=(opt,))

    # 启动两个线程
    t1.start()
    # choose_source = 1
    # t2.start()

    # 等待两个线程结束
    t1.join()
    # t2.join()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
