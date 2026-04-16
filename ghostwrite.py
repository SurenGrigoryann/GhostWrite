options = hand_landmarker.HandLandmarkerOptions(
    base_options=base_options.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionTaskRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

with hand_landmarker.HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
        mp_input = mp_image.Image(mp_image.ImageFormat.SRGB, frame_rgb)
        timestamp_ms = int(time.time() * 1000)

        results = landmarker.detect_for_video(mp_input, timestamp_ms)

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                fingertip_indices = [4, 8, 12, 16, 20]
                for idx in fingertip_indices:
                    lm = hand_landmarks[idx]
                    cx = int(lm.x * frame.shape[1])
                    cy = int(lm.y * frame.shape[0])
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Camera", cv2.WND_PROP_VISIBLE) < 1:
            break

cap.release()
cv2.destroyAllWindows()