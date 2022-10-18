from pathlib import Path
from typing import List
from retinaface import RetinaFace
import cv2
import sys

def convert(src :str, dst :str):
    cap = cv2.VideoCapture(src)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    f_s = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/4), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/4))
    f_d = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*4), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(dst, fourcc, cap.get(cv2.CAP_PROP_FPS), f_d)
    print(src, dst, fourcc, cap.get(cv2.CAP_PROP_FPS), frame_size)
    while cap.isOpened():
        ret, frame_src = cap.read()
        if frame_src is None:
            break
        frame_d = cv2.resize(frame_src, f_s, interpolation=cv2.INTER_LINEAR)
        resp = RetinaFace.detect_faces(frame_d)
        # print(resp)
        if isinstance(resp,dict):
            for v in resp.values():
                area_d = v['facial_area']
                area = [x * 4 for x in area_d]
                # Extract the region of the image that contains the face
                face_image = frame_src[area[1]:area[3], area[0]:area[2]]

                # Blur the face image
                # face_image = cv2.GaussianBlur(face_image, (21, 21), 0)

                # Get input size
                height, width = area[3]-area[1], area[2]-area[0]

                # Desired "pixelated" size
                w, h = (8, 8)

                # Resize input to "pixelated" size
                temp = cv2.resize(face_image, (w, h), interpolation=cv2.INTER_LINEAR)

                # Initialize output image
                face_image = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

                # Put the blurred face region back into the frame image
                frame_src[area[1]:area[3], area[0]:area[2]] = face_image
                cv2.rectangle(frame_src, (area[0], area[1]), (area[2], area[3]), (255,0,0), 10)
        frame_dst = cv2.resize(frame_src, f_d, interpolation=cv2.INTER_LINEAR)
        out.write(frame_dst)
        # cv2.imshow('video', frame_src)
        cv2.waitKey(1)

    cap.release()
    out.release()

def filels(src :str) -> List[str]:
    with open(src,'r') as f:
        return f.readlines()

def newdst(src :str, dstdir :str) -> str:
    p = Path(src)
    n = p.stem
    d = Path(dstdir)
    newname = d.joinpath(n + '_qn.mp4')
    return str(newname)

if __name__ == '__main__':
    srcfile = sys.argv[1]
    dstdir = sys.argv[2]
    for s in filels(srcfile):
        s = s.replace('\n', '').replace('\r', '')
        d = newdst(s, dstdir)
        convert(s, d)
