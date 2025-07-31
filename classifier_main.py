import serial
import time
import cv2
import tensorflow as tf
import numpy as np

# --- 1. 모델 로드 ---
# 'final_model.keras' 파일이 이 스크립트와 같은 디렉토리에 있어야 합니다.
try:
    model = tf.keras.models.load_model('final_model.keras')
    print("Keras model 'final_model.keras' loaded successfully.")
except Exception as e:
    print(f"Error loading Keras model: {e}")
    print("Please ensure 'final_model.keras' is in the correct path and TensorFlow is installed correctly.")
    exit()

# --- 2. 시리얼 포트 설정 ---
# 이 부분을 당신의 아두이노 포트와 일치시켜야 합니다.
SERIAL_PORT = '/dev/ttyACM0' # <-- 이 부분을 당신의 아두이노 포트 이름으로 수정하세요 (예: '/dev/ttyUSB0')
BAUD_RATE = 9600 # 아두이노 스케치의 Serial.begin() 값과 동일해야 합니다.

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) 
    time.sleep(2) # 시리얼 포트 초기화 대기 시간 (필수)
    print(f"Serial port {SERIAL_PORT} opened successfully at {BAUD_RATE} baud.")
except serial.SerialException as e:
    print(f"Error opening serial port {SERIAL_PORT}: {e}")
    print("Please ensure Arduino is connected, port name is correct, and you have necessary permissions.")
    exit()

# --- 3. 아두이노로 명령어 전송 함수 ---
def send_command_to_arduino(material):
    """분류 결과에 따라 아두이노로 명령을 전송합니다."""
    try:
        if material == "plastic":
            ser.write(b'P') # 'P' 문자열을 바이트 형태로 전송
            print("Sent 'P' for Plastic to Arduino.")
        elif material == "paper":
            ser.write(b'J') # 'J' 문자열을 바이트 형태로 전송
            print("Sent 'J' for Paper to Arduino.")
        else:
            print(f"Unknown material: '{material}'. No command sent.")
    except serial.SerialException as e:
        print(f"Error writing to serial port: {e}. Connection might be lost.")

# --- 4. 카메라 설정 및 메인 분류 루프 ---
def main_classification_loop():
    # IP Webcam 앱에서 확인한 스트림 URL로 변경합니다.
    # 예: 'http://192.168.43.123:8080/video' (IP Webcam 앱에 표시된 URL 뒤에 /video 또는 /live 등을 붙여야 할 수 있습니다.)
    # 정확한 URL은 IP Webcam 앱 화면에 표시됩니다.
    camera_stream_url = 'http://192.168.230.211:8080' # <-- 여기에 실제 URL 입력

    cap = cv2.VideoCapture(camera_stream_url)

    if not cap.isOpened():
        print(f"Error: Could not open camera stream at {camera_stream_url}. Please check the URL and network connection.")
        return

    print("Camera initialized. Press 'Ctrl+C' in the terminal to quit.") # 종료 안내 메시지 수정

    # 모델의 입력 이미지 크기 (final_model.keras의 InputLayer에서 확인)
    IMG_HEIGHT = 180
    IMG_WIDTH = 180

    try:
        while True:
            ret, frame = cap.read() # 카메라에서 프레임 읽기
            if not ret:
                print("Failed to grab frame. Exiting...")
                break

            # --- 이미지 전처리 (모델의 입력 요구사항에 맞게) ---
            # 1. 크기 조정: 모델의 입력 크기(180x180)에 맞게 조정
            resized_frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
            
            # 2. 차원 확장: 모델은 배치(batch) 형태로 입력을 받으므로 차원을 추가 (예: (1, 180, 180, 3))
            input_image = np.expand_dims(resized_frame, axis=0)
            
            # 3. 데이터 타입 변환 및 정규화: 0-1 사이의 float32 값으로 정규화 (모델 학습 시 정규화했다면 필수)
            input_image = tf.cast(input_image, tf.float32) / 255.0

            # --- 분류 모델 실행 (추론) ---
            predictions = model.predict(input_image)

            # 재질 분류 결과만 가져옵니다.
            material_predictions = predictions['material_output']
            
            # --- 예측 결과 해석 ---
            # material_predictions[0]은 [종이_확률, 플라스틱_확률] 형태의 배열이 됩니다.
            # 위에서 확인한 material_type_map에 따라: 'paper': 0, 'plastic': 1
            class_names_material = ["paper", "plastic"]
            # np.argmax를 사용하여 material_predictions[0] 배열에서 가장 높은 확률의 인덱스를 찾습니다.
            predicted_class_index_material = np.argmax(material_predictions[0])
            predicted_material = class_names_material[predicted_class_index_material]
            
            # 참고: object_output에 대한 예측도 필요하다면 유사하게 처리할 수 있습니다.
            # object_predictions = predictions['object_output']
            # object_type_map = {'cup': 0, 'cupholder': 1, 'lid': 2, 'straw': 3}
            # class_names_object = ["cup", "cupholder", "lid", "straw"]
            # predicted_class_index_object = np.argmax(object_predictions[0])
            # predicted_object = class_names_object[predicted_class_index_object]
            
            print(f"Predicted material: {predicted_material} (Probabilities: {material_predictions[0]})")

            # --- 분류 결과에 따라 아두이노로 명령 전송 ---
            send_command_to_arduino(predicted_material)
            
            # 서보 모터가 움직이고 물품이 지나갈 충분한 시간을 줍니다. (필요에 따라 조절)
            time.sleep(2) 

            # CLI 환경에서 프로그램을 종료하기 위한 키 입력 대기 (선택 사항)
            # 이 부분은 CLI 환경에서 작동하지 않으므로 제거하거나 주석 처리하는 것이 좋습니다.
            # key = cv2.waitKey(1) & 0xFF
            # if key == ord('q'): # 'q' 키를 누르면 종료
            #     print(" 'q' pressed. Exiting program.")
            #     break

    except KeyboardInterrupt:
        print("Program interrupted by user (Ctrl+C). Exiting.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # 프로그램 종료 시 카메라와 시리얼 포트 자원 해제
        if 'cap' in locals() and cap.isOpened():
            cap.release()
            print("Camera released.")
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port closed.")
        # cv2.destroyAllWindows() # CLI 환경에서는 GUI 창이 없으므로 필요 없음

if __name__ == "__main__":
    main_classification_loop()