import cv2
import os
import torch
import numpy as np
from torchvision import transforms
import torch.backends.cudnn as cudnn
from PIL import Image
from Retinaface.models.retinaface import RetinaFace
from Retinaface.data import cfg_mnet, cfg_re50
from Retinaface.utils.box_utils import decode, decode_landm

from Retinaface.layers.functions.prior_box import PriorBox
from Retinaface.utils.nms.py_cpu_nms import py_cpu_nms

from sixdrepnet.model import SixDRepNet
from sixdrepnet.utils import compute_euler_angles_from_rotation_matrices, draw_axis, plot_pose_cube

import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 加载 RetinaFace 模型
def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

transformations = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])



def get_model(netpath,modelpath,device):
    torch.set_grad_enabled(False)
    cfg = cfg_re50
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, netpath, False)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device(device)
    net = net.to(device)

    model = SixDRepNet(backbone_name='RepVGG-B1g2',
                       backbone_file='',
                       deploy=True,
                       pretrained=False)
    print('Loading data.')
    # Load snapshot
    snapshot_path = modelpath
    saved_state_dict = torch.load(os.path.join(
        snapshot_path), map_location='cpu')
    if 'model_state_dict' in saved_state_dict:
        model.load_state_dict(saved_state_dict['model_state_dict'])
    else:
        model.load_state_dict(saved_state_dict)
    model.to(device)
    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    return net, model

def face_detection(frame, net, device):
    img = np.float32(frame)

    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    start = time.time()
    loc, conf, landms = net(img)  # forward pass
    end = time.time()
    print('face detection time: %2f ms' % ((end - start)*1000.))

    resize = 1
    cfg = cfg_re50
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    confidence_threshold = 0.02
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    top_k = 5000
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    nms_threshold = 0.4
    keep = py_cpu_nms(dets, nms_threshold)

    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    keep_top_k = 750
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)

    return dets

def get_pose(frame,model, b, device):

    text = "{:.4f}".format(b[4])
    box = list(map(int, b))
    x_min = int(box[0])
    y_min = int(box[1])
    x_max = int(box[2])
    y_max = int(box[3])
    bbox_width = abs(x_max - x_min)
    bbox_height = abs(y_max - y_min)

    x_min = max(0, x_min-int(0.2*bbox_height))
    y_min = max(0, y_min-int(0.2*bbox_width))
    x_max = x_max+int(0.2*bbox_height)
    y_max = y_max+int(0.2*bbox_width)

    img = frame[y_min:y_max, x_min:x_max]
    img = Image.fromarray(img)
    img = img.convert('RGB')
    img = transformations(img)

    img = torch.Tensor(img[None, :]).to(device)

    start = time.time()
    R_pred = model(img)
    end = time.time()
    print('HPE estimation time: %2f ms' % ((end - start)*1000.))

    euler = compute_euler_angles_from_rotation_matrices(
        R_pred)*180/np.pi
    p_pred_deg = euler[:, 0].cpu()
    y_pred_deg = euler[:, 1].cpu()
    r_pred_deg = euler[:, 2].cpu()
    #draw_axis(frame, y_pred_deg, p_pred_deg, r_pred_deg, left+int(.5*(right-left)), top, size=100)
    plot_pose_cube(frame,  y_pred_deg, p_pred_deg, r_pred_deg, x_min + int(.5*(
            x_max-x_min)), y_min + int(.5*(y_max-y_min)), size=bbox_width)
    return frame , p_pred_deg


if __name__ == '__main__':
    device = "cuda"
    net, model = get_model(device)
    # 打开本地摄像头
    cap = cv2.VideoCapture(0)
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()
    # 实时检测人脸
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头帧")
            break

        dets = face_detection(frame,net,device)

        # show image
        for b in dets:
            vis_thres = 0.6
            if b[4] < vis_thres:
                continue

            frame = get_pose(frame,model,b,device)

        # 显示结果
        cv2.imshow('Face Detection', frame)

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头资源并关闭窗口
    cap.release()
    cv2.destroyAllWindows()