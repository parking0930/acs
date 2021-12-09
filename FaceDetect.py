import os
import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
import picamera
import logging
import telegram
from os import listdir
from os.path import isdir, isfile, join
from datetime import datetime
# 도어락 제어 GPIO, 14번 핀 사용 프로그램 실행시, 잠겨있는 상태가 Defalut가 되도록
GPIO.setmode(GPIO.BCM)
GPIO.setup(14, GPIO.OUT, initial=GPIO.HIGH)
bot = telegram.Bot(token='2063832595:AAHXOcez_70VveYrHl9c7xoRJuJ7SphoRAo')  # 텔레그램 봇 연결 키
chat_id = 2078275327
# 얼굴 인식용 haarcascade 로딩
face_classifier = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')    

# 사용자 얼굴 학습
def train(name):
    data_path = 'faces/' + name + '/'
    #파일만 리스트로 만듬
    face_pics = [f for f in listdir(data_path) if isfile(join(data_path,f))]
    
    Training_Data, Labels = [], []
    
    # 사용자의 사진을 불러옴
    for i, files in enumerate(face_pics):
        image_path = data_path + face_pics[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # 이미지가 아니면 패스
        if images is None:
            continue    
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)
    # 학습할 데이터(이미지)가 없으면 함수 종료
    if len(Labels) == 0:
        print("학습할 이미지가 없습니다.")
        return None
    # Lables 를 32비트 정수로 변환함
    Labels = np.asarray(Labels, dtype=np.int32)
    # 모델 생성
    model = cv2.face.LBPHFaceRecognizer_create()
    # 학습
    model.train(np.asarray(Training_Data), np.asarray(Labels))
    print(name + " : 모델 학습 완료")

    #학습 모델 리턴
    return model

# 여러 사용자 학습
def trains():
    #faces 폴더의 하위 폴더를 학습
    data_path = 'faces/'
    # 폴더만 색출
    model_dirs = [f for f in listdir(data_path) if isdir(join(data_path,f))]
    
    #학습 모델 저장할 딕셔너리
    models = {}
    # 각 폴더에 있는 얼굴들 학습
    for model in model_dirs:
        print('model :' + model)
        # 학습 시작
        result = train(model)
        # 학습이 안되었다면 패스!
        if result is None:
            continue
        # 학습되었으면 저장
        print('model2 :' + model)
        models[model] = result

    # 학습된 모델 딕셔너리 리턴
    return models   

#얼굴 검출
def face_detector(img, size = 0.5):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)
        if faces is():
            return img,[]
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200,200))
        return img,roi   #검출된 좌표에 사각 박스 그리고(img), 검출된 부위를 잘라(roi) 전달

# 인식 시작
def run(models):    
    #카메라 열기 
    cap = cv2.VideoCapture(0)
    try:
        while True:
            #카메라로 부터 사진 한장 읽기 
            ret, frame = cap.read()
            # 얼굴 검출 시도 
            image, face = face_detector(frame)
            try:            
                min_score = 999       #예측된 사람일 확률
                min_score_name = ""   #예측된 사람의 이름
            
                #검출된 사진을 흑백으로 변환 
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                #위에서 학습한 모델로 예측시도
                for key, model in models.items():
                    result = model.predict(face)                
                    #result[1] 에 들어가는 값은 신뢰도 임
                    if min_score > result[1]:
                        min_score = result[1]
                        min_score_name = key
                    
                # 0에 가까울 수록 정확하다.         
                if min_score < 500:
                    # 정확도 
                    confidence = int(100*(1-(min_score)/300))
                    # 유사도 화면에 표시 
                    display_string = str(confidence)+'% Confidence it is ' + min_score_name
                cv2.putText(image,display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)
                
                #80 보다 크면 동일 인물로 간주해 UnLocked! 
                if confidence > 80:
                    cv2.putText(image, "Unlocked : " + min_score_name, (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Face Cropper', image)
                    
                    # 저장할 파일의 경로와 이름지정
                    now = str(datetime.now().strftime('%Y-%m-%d %H:%M'))
                    file_list = os.listdir('./Save')
                    # 중복된 사진이 있는지 검사용 변수
                    exist_person = False
                    for filename in file_list:
                        if filename == (min_score_name+'|'+now+'.jpg'):
                            print("이미 있는 사진")
                            exist_person = True
                            break
                    if exist_person:
                        continue
                    # 이름_사원번호를 _ 기준으로 분리
                    p_name = min_score_name.split('_')[0]
                    uid = min_score_name.split('_')[1]
                    # 인증된 사용자이므로 도어락 열기
                    GPIO.output(14, GPIO.LOW)
                    #텔레그램으로 사진 전송 (인증된 사용자)
                    cap_name = min_score_name+'|'+now + ".jpg"
                    cv2.imwrite('./Save/'+cap_name, frame)
                    bot.sendPhoto(chat_id=chat_id,photo=open('./Save/'+cap_name,'rb'))
                    log_text = '사원번호:'+uid+'\n이름:'+p_name
                    bot.send_message(chat_id=chat_id, text=log_text) 
                
                #80 이하면 타인.. Locked!!!
                else:
                    for filename in file_list:
                        if filename == ('X|'+now+'.jpg'):
                            print("이미 있는 사진")
                            exist_person = True
                            break
                    if exist_person:
                        continue
                    #텔레그램으로 사진 전송 (비인가 사용자)
                    cap_name = 'X|'+now + ".jpg"
                    cv2.imwrite('./Save/'+cap_name, frame)
                    bot.sendPhoto(chat_id=chat_id,photo=open('./Save/'+cap_name,'rb'))
                    log_text = '인가되지 않은 사용자 발견!'
                    bot.send_message(chat_id=chat_id, text=log_text) 
                    cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('Face Cropper', image)

            #얼굴 검출 안됨 
            except:
                cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow('Face Cropper', image)
                pass
            #Enter 입력으로 프로그램 종료
            if cv2.waitKey(1)==13:
                break
        #카메라 끄기
        cap.release()
        #열려있는 창 닫기
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        GPIO.cleanup()


if __name__ == "__main__":
    GPIO.output(14, GPIO.HIGH)
    # 학습 시작
    models = trains()
    # 시작
    run(models)
