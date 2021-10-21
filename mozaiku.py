import cv2 as cv


# setting
face_cascade_name = 'env/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml'
eyes_cascade_name = 'env/lib/python3.7/site-packages/cv2/data/haarcascade_eye.xml'

def mosaic(src, ratio=0.1):
    small = cv.resize(src, None, fx=ratio, fy=ratio, interpolation=cv.INTER_NEAREST)
    return cv.resize(small, src.shape[:2][::-1], interpolation=cv.INTER_NEAREST)

def mosaic_area(src, x, y, width, height, ratio=0.1):
    dst = src.copy()
    dst[y:y + height, x:x + width] = mosaic(dst[y:y + height, x:x + width], ratio)
    return dst



def detectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    # 顔の検出
    faces = face_cascade.detectMultiScale(frame_gray)
    # 複数の顔が検出された場合、ひとつづつ枠を付ける
    for (x,y,w,h) in faces:
        print('detected')
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        
        faceROI = frame_gray[y:y+h,x:x+w]

        # 顔ごとに目を検出する
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))


            ex = int((x + x2) - w2 / 2)
            ey = int(y + y2)
            ew = int(w2 * 2.5)
            frame = mosaic_area(frame, ex, ey, ew, h2)
            #frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), -1)


    #cv.imshow('OpenCV - facedetect', frame)
    cv.imshow('OpenCV - test', frame_gray)


if __name__ == "__main__":
    # 物体検出用のインスタンス作成
    face_cascade = cv.CascadeClassifier()
    eyes_cascade = cv.CascadeClassifier()

    face_cascade.load(cv.samples.findFile(face_cascade_name))
    eyes_cascade.load(cv.samples.findFile(eyes_cascade_name))

    # cascadesファイルのロード
    if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
        print('--(!)Error loading face cascade')
        exit(0)
    if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
        print('--(!)Error loading eyes cascade')
        exit(0)

    # キャプチャスタート
    cap = cv.VideoCapture(0)
    try:   
        while True:
            ret, frame = cap.read()
            if frame is None:
                print('--(!) No captured frame -- Break!')
                break

            # ビデオ上にテキストを表示 (カメラデータ, 文字, (表示位置), フォント, フォントサイズ, 色, 太さ, 線の種類)
            cv.putText(frame, 'mokemoke', (200,50), cv.FONT_HERSHEY_PLAIN, 3, (0, 255,0), 3, cv.LINE_AA)

            detectAndDisplay(frame)

            if cv.waitKey(10) == 27:
                break

    except KeyboardInterrupt: # except the program gets interrupted by Ctrl+C on the keyboard.
        print("\nCamera Interrupt")

    finally:
        cap.release()
        cv.destroyAllWindows()