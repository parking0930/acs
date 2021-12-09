# 2021-2 무선네트워크 프로젝트 출입 관리 시스템
## 프로젝트 소개
<table>
    <tr>
        <th>프로젝트 명 </th>
        <th>출입 관리 시스템</th>
        <th>개발기간</th>
        <th>2021.11.12 ~ 2021.12.09</th>
    </tr>
    <tr>
        <th>프로젝트 성격</th>
        <th>기말고사</th>
        <th>개발인원</th>
        <th>팀 / 3명<br></th>
    </tr>
      <tr>
        <th>프로젝트 개요</th>
        <th>라즈베리파이 카메라를 이용한 출입 관리 시스템</th>
        <th>개발환경&nbsp;</th>
        <th>Raspberry Pi</th>
    </tr>
    <tr>
        <th colspan="1">사용 도구</th>
        <th colspan="3">카메라 모듈, 릴레이 모듈, 도어락</th>
    </tr>
    <tr>
        <th>개발언어</th>
        <th colspan="3">Python, Spring Framework, MySQL</th>
    </tr>
    <tr>
        <th>활용 라이브러리</th>
        <th colspan="3">OpenCV</th>
    </tr>
    <tr>
        <th>DB 서버</th>
        <th colspan="3">https://github.com/parking0930/acs_server</th>
    </tr>
    <tr>
        <th>시연 영상</th>
        <th colspan="3">
          https://www.youtube.com/watch?v=39EuufjgfME
        </th>
    </tr>
</table>

### 📑 역할 분담
<table>
    <tr>
        <th width="20%">업무 / <br>구성원</th>
        <th width="25%">모영훈</th>
        <th width="25%">박재현</th>
        <th width="25%">이강민</th>
    </tr>
    <tr>
        <th>프로젝트 기획</th>
        <th colspan="3">개요작성,회의,의견제안</th>
    </tr>
    <tr>
        <th rowspan=2>역할</th>
        <th>OpenCV 코드 설계<br>카메라 모듈<br>도어락 연결<br>DB 서버 연동</th>
        <th>DB 서버 구축(Spring)<br>소스 취합<br>텔레그램 연동<br>OpenCV 코드 설계</th>
        <th>DB 서버 연동<br> 카메라 모듈<br> 도어락 연결<br> 텔레그램 연동<br>OpenCV 코드 설계</th>
    </tr>
    <tr>
        <th colspan=3>자료 조사, 버그 수정, 발표자료 작성</th>
    </tr>
</table>

### - 출입 관리 시스템
  - 사람 출입 시 텔레그램으로 알림(등록 여부, 등록 정보 등)
  - 출입 시마다 데이터베이스에 출입 시간, 출입자 정보 등을 기록
  - 사용자의 이름, 사진 등의 정보를 사전에 테이블에 등록

## 개발 목표
### - 프로젝트 계획
  - python opencv 라이브러리 또는 별도의 API 활용
  - 출입 기록을 저장할 데이터베이스 테이블 생성

### - 실행 과정
    1. 사전에 사용자 사진 등록하여 학습
    2. 인체감지(PIR) 센서로 사람 감지 시 카메라를 켜고 사진 촬영
    3. 등록(인가)된 사용자인지 판별(opencv 활용)
    4. 등록된 사용자 여부, 사용자 정보, 촬영된 사진을 관리자 텔레그램으로 전송
    5. 출입 관련 데이터(사용자명, 인가 여부, 출입 시간) 테이블에 기록
    
## 환경 구축
### - OpenCV설치
    1. 패키지 업데이트
    sudo apt-get upgrade
    
    2. 패키지 설치
    sudo apt-get install build-essential cmake -y
    sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev -y
    sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libxvidcore-dev libx264-dev libxine2-dev -y
    sudo apt-get install libv4l-dev v4l-utils -y
    sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev -y
    sudo apt-get install libgtk2.0-dev -y
    sudo apt-get install mesa-utils libgl1-mesa-dri libgtkgl2.0-dev libgtkglext1-dev -y
    sudo apt-get install libatlas-base-dev gfortran libeigen3-dev -y
    sudo apt-get install python2.7-dev python3-dev python-numpy python3-numpy -y
  
    3. OpenCV 다운할 폴더 생성
    mkdir opencv
    
    4. 폴더 이동 
    cd opencv
    
    5. 다운 및 압축해제
    wget -O opencv.zip https://github.com/opencv/opencv/archive/4.1.2.zip
    unzip opencv.zip
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.1.2.zip
    unzip opencv_contrib.zip
    
    6. 폴더 이동 및 컴파일
    cd opencv-4.1.2
    
    7. 폴더 이동 -> cd build
    
    8. OpenCV 컴파일 설정
    
    9. 주기억 장치 메모리 설정 100 -> 2048
    sudo nano /etc/dphys-swapfile
    sudo /etc/init.d/dphys-swapfile restart
    
    10. 컴파일
    make -j4
    
    11. 설치
    sudo make install
    
    12. openCV라이브러리를 찾을 수 있도록 설정
    sudo ldconfig
    
    13. 메모리 설정 되돌리기
    sudo nano /etc/dphys-swapfile
    
    14. 설정 파일 재시작
    sudo /etc/init.d/dphys-swapfile restart
    
    15. OpenCV는 Haar Feature를 설치를 한다. 
### - 학습 알고리즘(Haar Cascade)
Haar Cascade의 특징은 계산을 통해 객체를 찾아서 사진에서 어두운 부분과 밝은 부분의 차이를 통해 사진의 특징을 학습하는 방식, 그렇기 때문에 회색으로 변환된 사진을 이용하면 연산의 양을 줄여줄 수 있다.<br>
![그림5](https://user-images.githubusercontent.com/28342911/145423205-3cccc8ac-4488-47c7-9650-92fe93ec660a.png)
![그림6](https://user-images.githubusercontent.com/28342911/145423208-c43de879-e3d1-423b-b422-7b7c76562ea7.png)
<br/>
왼쪽이 Haar CasCade 의 특징 검출 방식이고, 오른쪽이 실 적용 모습이다. 눈에 눈썹이나 동공같이 검은부분은 검은 사각형으로 검출되고, 피부는 비교적 밝으므로 하얀색으로 검출된다.
Cascade 분류기를 사용한 얼굴 검출은 얼굴이 가려지거나 하면 인식률이 현저히 떨어지기 때문에 얼굴 인식에는 딥러닝 기반 얼굴 검출이 더 좋다.

### - Haar feature-based cascade 선정 이유
 - opencv 버전 문제, 라즈베리파이 성능 문제를 고려해보았을 때 Haar Cascade가 가장 적합한 알고리즘이었음.
<table>
  <tr>
    <td>
      <img src="https://user-images.githubusercontent.com/28342911/145423152-8b212600-e1ed-4166-affb-923133734bc3.jpg" style="width:450px;display:inline-block;">
    </td>
    <td>
      <img src="https://user-images.githubusercontent.com/28342911/145423117-bfd152fe-6747-4e89-adb2-29f099aedc89.jpg" style="width:450px;display:inline-block;">
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://user-images.githubusercontent.com/28342911/145423088-af508b07-d419-4d1f-bfda-fca07c4364c4.jpg" style="width:450px;display:inline-block;">
    </td>
    <td>
      <img src="https://user-images.githubusercontent.com/28342911/145423141-217e29e1-cd9a-4cec-9046-7e51761a139d.jpg" style="width:450px;display:inline-block;">
    </td>
  </tr>
</table>


## 개발
- 사용자 등록 (FaceRegister.py)
<details mardown="1">
    <summary> FaceRegister.py ① </summary>
  
~~~python
# 얼굴을 저장할 디렉토리
face_dirs = 'faces/'
# haarcascade 학습모듈 로드
face_classifier = cv2.CascadeClassifier('/home/pi/pj/haarcascades/haarcascade_frontalface_default.xml')
~~~
  
</details>
<details mardown="1">
    <summary> FaceRegister.py ② </summary>
  
~~~python
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
~~~
 
</details>
<br>
- 출입 관리 시스템 (FaceDetect.py)
<details mardown="1">
    <summary> FaceDetect.py ① </summary>
  
~~~python
# 도어락 제어 GPIO, 14번 핀 사용 프로그램 실행시, 잠겨있는 상태가 Defalut가 되도록
GPIO.setmode(GPIO.BCM)
GPIO.setup(14, GPIO.OUT, initial=GPIO.HIGH)
bot = telegram.Bot(token='2063832595:AAHXOcez_70VveYrHl9c7xoRJuJ7SphoRAo')  # 텔레그램 봇 연결 키
chat_id = 2078275327
# 얼굴 인식용 haarcascade 로딩
face_classifier = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml') 
~~~
  
</details>
<details mardown="1">
    <summary> FaceDetect.py ② </summary>
  
~~~python
if __name__ == "__main__":
    # 학습 시작
    models = trains()
    # 시작
    run(models)
~~~
 
</details>

</details>
<details mardown="1">
    <summary> FaceDetect.py ③ </summary>
  
~~~python
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
~~~
 
</details>
<details mardown="1">
    <summary> FaceDetect.py ④ </summary>
  
~~~python
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
~~~
 
</details>
<details mardown="1">
    <summary> FaceDetect.py ⑤ </summary>
  
~~~python
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
~~~
 
</details>
<details mardown="1">
    <summary> FaceDetect.py ⑤ </summary>
  
~~~python
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
~~~
 
</details>
    
## 한계점
- 텔레그램으로 인가 여부 및 정보를 보냈으나 DB로 보내는 부분은 문제가 생겨 하지 못하였음. 
![1](https://user-images.githubusercontent.com/28342911/145446239-44bcf6ef-4460-4191-af80-d169d3946507.JPG)
