from tf_bodypix.api import download_model, load_model
import cv2

bodypix_model = load_model(download_model('resnet50'))

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = bodypix_model.predict_single(rgb_frame)
    mask = result.get_mask(threshold=0.7).astype('uint8') * 255

    silhouette = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # Optional: Silhouette weiß, Rest schwarz
    output = silhouette.copy()

    cv2.imshow("BodyPix Silhouette", output)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
