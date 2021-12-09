import cv2
import numpy as np
import requests
import json
from os import makedirs
from os.path import isdir

# 얼굴을 저장할 디렉토리
face_dirs = 'faces/'
# haarcascade 학습모듈 로드
face_classifier = cv2.CascadeClassifier('/home/pi/pj/haarcascades/haarcascade_frontalface_default.xml')

# 얼굴 검출 함수
def face_extractor(img):
    # 사진을 회색으로 변환 (연산량을 줄일 수 있음)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    # 얼굴이 없으면 넘어감
    if faces is():
        return None
    # 얼굴이 있으면 얼굴 부위만 이미지로 만들고
    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]
    # 자른 이미지를 반환
    return cropped_face

# 사진 촬영 함수
def take_pictures(name):
    # 해당 이름의 폴더가 없다면 생성
    if not isdir(face_dirs+name):
        makedirs(face_dirs+name)

    # 카메라 ON    
    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        # 카메라로 부터 사진 한장 읽어 오기
        ret, frame = cap.read()
        # 사진에서 얼굴 검출 , 얼굴이 검출되었다면 
        if face_extractor(frame) is not None:
            count+=1
            # 촬영한 사진을 200 x 200 사이즈로 조정
            face = cv2.resize(face_extractor(frame),(200,200))
            # 흑백으로 바꿈
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            # 200x200 흑백 사진을 faces/얼굴 이름/userxx.jpg 로 저장
            file_name_path = face_dirs + name + '/' + str(count) +'.jpg'
            cv2.imwrite(file_name_path,face)
            # 촬영된 사진이 몇번째 사진인지 창 안에 표시
            cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow('Detected Face',face)
        else:
            print("Face not Found")
            pass
        
        # 얼굴 사진 30장을 촬영했거나 enter키 누르면 종료
        if cv2.waitKey(1)==13 or count==30:
            break

    # 촬영 종료
    cap.release()
    # 창 닫음
    cv2.destroyAllWindows()
    print('학습 샘플이미지 추출 완료')

if __name__ == "__main__":
    p_number = ""
    # 사진 저장할 이름을 넣어서 함수 호출
    while True:
        p_number = input("사원번호 : ")
        request_data = requests.get("http://172.20.10.5:8081/infoserver/selectAuthorized?id="+p_number)
        info = json.loads(request_data.text)
        if info["authorized"] == None:
            break
        print("이미 존재하는 사원번호입니다.")
    # print(result)

    name = input("학습할 사람의 이름 : ")
    position = input("직책 : ")
    take_pictures(name + "_" + p_number)

    response_data = {"id" : p_number,
                "name" : name,
                "position": position
                }
    result = requests.post("http://172.20.10.5:8081/infoserver/register", data=response_data)
