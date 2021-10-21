import cv2
import time
import requests
from datetime import datetime

movie = cv2.VideoCapture(0)

red = (0, 0, 255) # 枠線の色
before = None # 前回の画像を保存する変数
fps = int(movie.get(cv2.CAP_PROP_FPS)) #動画のFPSを取得
pre_location = 0
count = 0
detected = 0

def sendMessage(num, count, path):
    url = "https://notify-api.line.me/api/notify" 
    token = "bwdwyHu7TTye4bDLtfoFdvXqhiQQw6KqOCRj7IzejLE"
    headers = {"Authorization" : "Bearer "+ token} 
    if num > 0:
        m = "\n" + str(num) + "人入室しました。"
    else:
        m = "\n" + str(-1*num) + "人退室しました。"
    message = m + "\n現在の人数は" + str(count) + "人です。\nみちょぱは元気です。"
    payload = {"message" :  message} 
    files = {"imageFile": open(path, "rb")} 
    r = requests.post(url, headers = headers, params=payload, files=files) 

def takePic(ret, frame):
    if ret:
        # ファイル名に日付を指定
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = "./images/" + date + ".jpg"
        cv2.imwrite(path, frame)
    return path

while True:
    # 画像を取得
    ret, frame = movie.read()
    # 再生が終了したらループを抜ける
    if ret == False: break
    # 白黒画像に変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if before is None:
        before = gray.astype("float")
        continue
    #現在のフレームと移動平均との差を計算
    cv2.accumulateWeighted(gray, before, 0.8)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(before))
    #frameDeltaの画像を２値化
    thresh = cv2.threshold(frameDelta, 3, 255, cv2.THRESH_BINARY)[1]
    #輪郭のデータを得る
    contours = cv2.findContours(thresh,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)[0]

    location = 0
    valid = 0
    valid_move = 0
    # 差分があった点を画面に描く
    for target in contours:
        x, y, w, h = cv2.boundingRect(target)
        if w < 100: continue # 小さな変更点は無視
        valid = 1
        cv2.rectangle(frame, (x, y), (x+w, y+h), red, 2)
        location += x + w/2
        valid_move += 1    

    if valid == 1:
        detected += 1
        location /= valid_move
        #If nothing is detected in the previous frame
        if detected == 1:
            pre_location = location

        if detected == 10:
            if location > pre_location:
                count += 1
                path = takePic(ret, frame)
                sendMessage(1, count, path)
            elif location < pre_location:
                count -= 1
                path = takePic(ret, frame)
                sendMessage(-1, count, path)
    else:
        detected = 0
        
    # ビデオ上にテキストを表示 (カメラデータ, 文字, (表示位置), フォント, フォントサイズ, 色, 太さ, 線の種類)
    cv2.putText(frame, str(count)+"ppl", (250,450), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "in", (550,400), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255,0), 2, cv2.LINE_AA)
    cv2.putText(frame, "out", (50,400), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255,0), 2, cv2.LINE_AA)

    #ウィンドウでの再生速度を元動画と合わせる
    time.sleep(1/fps)
    # ウィンドウで表示
    cv2.imshow('target_frame', frame)
    # Enterキーが押されたらループを抜ける
    if cv2.waitKey(1) == 13: break

movie.release()
cv2.destroyAllWindows() # ウィンドウを破棄