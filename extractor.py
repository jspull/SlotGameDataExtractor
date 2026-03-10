import cv2
import easyocr
import torch
import traceback
import re
import time
import collections
import numpy as np
from PIL import Image
from PyQt5.QtCore import QThread, pyqtSignal

# ── CLIP 이벤트 라벨 기본값 ──
DEFAULT_EVENT_ENTRIES = [
    {"name": "JACKPOT",    "prompt": "slot machine jackpot winning screen with big text"},
    {"name": "SUPER WIN",  "prompt": "slot machine super win celebration with flashing effects"},
    {"name": "BIG WIN",    "prompt": "slot machine big win announcement on screen"},
    {"name": "MEGA WIN",   "prompt": "slot machine mega win display with coins"},
    {"name": "FREE SPIN",  "prompt": "slot machine free spin bonus round starting"},
    {"name": "BONUS GAME", "prompt": "slot machine bonus game special screen"},
    {"name": "SCATTER",    "prompt": "slot machine scatter symbol activation"},
    {"name": "WILD",       "prompt": "slot machine wild feature triggered"},
    {"name": "RESPIN",     "prompt": "slot machine respin feature"},
]

# 마지막에 항상 추가되는 'normal gameplay' 라벨 (사용자 변경 불가)
NORMAL_LABEL = "normal slot machine gameplay with spinning reels"


class ExtractorThread(QThread):
    progress_signal = pyqtSignal(int)
    data_signal = pyqtSignal(int, str, float, float, float, str)  # +event_type
    status_signal = pyqtSignal(str)
    raw_log_signal = pyqtSignal(str) # 프레임 스캔 원본 데이터
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    elapsed_signal = pyqtSignal(str)

    def __init__(self, video_path, roi_bal, roi_win, roi_event=None, start_frame=0, 
                 clip_threshold=0.5, event_entries=None, bal_filter=None, fixed_bet=None,
                 roi_bal_filter=None, roi_win_filter=None, stability_pct=0.5,
                 drop_only_spin=False):
        super().__init__()
        self.video_path = video_path
        self.roi_bal = roi_bal
        self.roi_win = roi_win
        self.roi_event = roi_event
        self.start_frame = start_frame
        self.clip_threshold = clip_threshold
        self.bal_filter = bal_filter
        self.fixed_bet = fixed_bet
        self.roi_bal_filter = roi_bal_filter
        self.roi_win_filter = roi_win_filter
        self.stability_pct = stability_pct  # 안정 감지 임계값 (%)
        self.drop_only_spin = drop_only_spin  # Drop Only Spin 모드
        self._is_stopped = False
        
        # 동적 이벤트 라벨 구성
        entries = event_entries if event_entries else DEFAULT_EVENT_ENTRIES
        self._event_names = [e["name"] for e in entries] + [""]  # 마지막은 normal
        self._event_labels = [e["prompt"] for e in entries] + [NORMAL_LABEL]
        # CLIP 관련
        self._clip_model = None
        self._clip_processor = None
        self._clip_text_inputs = None

    def stop(self):
        self._is_stopped = True

    def _init_easyocr(self, device):
        """EasyOCR 리더 초기화"""
        reader = easyocr.Reader(['en'], gpu=(device == 'cuda'))
        return reader

    def _init_clip(self, device):
        """CLIP 모델 초기화 (Event ROI가 설정된 경우에만)"""
        if self.roi_event is None:
            return
        try:
            from transformers import CLIPProcessor, CLIPModel
            self.status_signal.emit("Loading CLIP model for event detection...")
            self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self._clip_model = self._clip_model.to(device)
            self._clip_model.eval()
            # 텍스트 프롬프트 사전 인코딩 (1회)
            self._clip_text_inputs = self._clip_processor(
                text=self._event_labels, return_tensors="pt", padding=True
            ).to(device)
        except Exception as e:
            self.status_signal.emit(f"CLIP init failed: {e}. Event detection disabled.")
            self.roi_event = None

    def _classify_event(self, frame, device, easy_reader):
        """CLIP 분류 및 OCR 교차 검증"""
        if self._clip_model is None or self.roi_event is None:
            return ""
        try:
            ex, ey, ew, eh = self.roi_event
            roi_frame = frame[ey:ey+eh, ex:ex+ew]
            rgb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            image_inputs = self._clip_processor(images=pil_img, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = self._clip_model(**image_inputs, **self._clip_text_inputs)
                probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
            top_idx = probs.argmax()
            normal_idx = len(self._event_labels) - 1
            
            if top_idx != normal_idx and probs[top_idx] >= self.clip_threshold:
                return self._event_names[top_idx]
            
            return ""
        except Exception:
            return ""

    def run(self):
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            gpu_msg = "GPU Accel" if device == 'cuda' else "CPU Mode"

            self.status_signal.emit(f"Initializing EasyOCR ({gpu_msg}). Please wait...")
            easy_reader = self._init_easyocr(device)

            # CLIP 초기화 (Event ROI 설정 시에만)
            self._init_clip(device)
            event_capture_frame = None  # ROLLING_UP 진입 시 프레임 캡처용

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error_signal.emit("Failed to open video file.")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0: fps = 30.0

            # FPS / 15: approx 15 dense checks per second
            frames_to_skip = int(fps / 15)
            if frames_to_skip < 1: frames_to_skip = 1
            rolling_stable_count = 0
            settling_stable_count = 0
            settling_time_sec = 0.0
            frame_idx = 0

            # Jump to start frame if specified
            if self.start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
                frame_idx = self.start_frame

            start_time = time.time()
            self.status_signal.emit(f"Processing Video (EasyOCR, {gpu_msg}). Extracting data...")

            # ROI 필터 적용 함수 (v44 포팅)
            def _apply_roi_filter(gray_img, filter_dict):
                """ROI에 밝기/대비/이진화 필터 적용"""
                if filter_dict is None or not filter_dict:
                    return gray_img
                img = gray_img.astype(np.float64)
                contrast = filter_dict.get("contrast", 100) / 100.0
                brightness = filter_dict.get("brightness", 0)
                img = img * contrast + brightness
                img = np.clip(img, 0, 255).astype(np.uint8)
                if filter_dict.get("threshold_on", 0):
                    bs = filter_dict.get("block_size", 11)
                    if bs % 2 == 0:
                        bs += 1
                    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bs, 2)
                return img

            # Balance ROI용: 사용자 필터 적용 → OCR
            def read_roi(roi, frame_img):
                x, y, w, h = roi
                roi_frame = frame_img[y:y+h, x:x+w]
                gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                scaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

                # 사용자 필터가 있으면 필터 적용, 없으면 Otsu 폴백
                if self.roi_bal_filter:
                    filtered = _apply_roi_filter(scaled, self.roi_bal_filter)
                else:
                    _, filtered = cv2.threshold(scaled, 120, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                def read_img(img):
                    res = easy_reader.readtext(img, detail=0)
                    return " ".join(res).strip() if res else ""

                text = read_img(filtered)
                if not text: text = read_img(scaled)
                if not text: text = read_img(gray)
                if not text: return None

                letters_count = sum(1 for c in text if c.isalpha())
                digits_count = sum(1 for c in text if c.isdigit())
                if letters_count > 0 and letters_count >= digits_count:
                    return None

                char_map = {'O':'0', 'o':'0', 'l':'1', 'i':'1', 'I':'1',
                            'S':'5', 's':'5', 'B':'8', 'Z':'2', 'z':'2', 'G':'6', 'g':'9'}
                text = "".join([char_map.get(c, c) for c in text])

                clean = re.sub(r'[^\d.]+', '', text)
                parts = clean.split('.')
                if len(parts) > 1:
                    last_part = parts[-1]
                    if len(last_part) == 3:
                        clean = "".join(parts)
                    else:
                        clean = "".join(parts[:-1]) + '.' + parts[-1]
                if not clean or clean == '.': return None
                try: return float(clean)
                except ValueError: return None

            # Win ROI용: 사용자 필터 적용 → OCR
            def read_roi_win(roi, frame_img):
                x, y, w, h = roi
                roi_frame = frame_img[y:y+h, x:x+w]
                gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                scaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

                # 사용자 필터가 있으면 필터 적용, 없으면 Otsu 폴백
                if self.roi_win_filter:
                    filtered = _apply_roi_filter(scaled, self.roi_win_filter)
                else:
                    _, filtered = cv2.threshold(scaled, 120, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                # filtered 이미지 1회만 시도
                res = easy_reader.readtext(filtered, detail=0)
                text = " ".join(res).strip() if res else ""
                if not text: return None

                letters_count = sum(1 for c in text if c.isalpha())
                digits_count = sum(1 for c in text if c.isdigit())
                if letters_count > 0 and letters_count >= digits_count:
                    return None

                char_map = {'O':'0', 'o':'0', 'l':'1', 'i':'1', 'I':'1',
                            'S':'5', 's':'5', 'B':'8', 'Z':'2', 'z':'2', 'G':'6', 'g':'9'}
                text = "".join([char_map.get(c, c) for c in text])

                clean = re.sub(r'[^\d.]+', '', text)
                parts = clean.split('.')
                if len(parts) > 1:
                    last_part = parts[-1]
                    if len(last_part) == 3:
                        clean = "".join(parts)
                    else:
                        clean = "".join(parts[:-1]) + '.' + parts[-1]
                if not clean or clean == '.': return None
                try: return float(clean)
                except ValueError: return None

            def update_seq(seq, val, max_len=5):
                seq.append(val)
                if len(seq) > max_len:
                    seq.pop(0)

            def get_stable(seq):
                if len(seq) == 5:
                    valid_vals = [x for x in seq if x is not None]
                    if len(valid_vals) >= 3:
                        counter = collections.Counter(valid_vals)
                        most_common_val, count = counter.most_common(1)[0]
                        if count >= 3:
                            return True, most_common_val
                return False, None

            # --- 2-Stage Pipeline (2초 딜레이) 데이터 버퍼 및 로직처리기 준비 ---
            raw_buffer = []  # 감지된 데이터를 쌓아둘 버퍼: (frame_idx, time_sec, raw_bal, raw_win, clip_event)
            fps_int = int(fps)
            logic_processor = LogicProcessor(fps_int, self.data_signal, self.bal_filter, self.fixed_bet, self.drop_only_spin)

            # --- absdiff 기반 변화 감지 변수 (v44 속도 기법 도입) ---
            # 이전 프레임의 흑백 ROI 이미지 보관용
            prev_bal_gray = None
            prev_win_gray = None
            # 안정 카운터: 화면이 안 변하고 있는 연속 프레임 수
            bal_stable_count = 0
            win_stable_count = 0
            # 마지막으로 OCR이 성공적으로 읽어낸 값 보관
            last_ocr_bal = None
            last_ocr_win = 0.0
            # 안정 판정 임계값: 처리 프레임 기준 ~0.3초 연속 안정 시 OCR 1회 호출
            stability_threshold = max(3, int(0.3 * fps / max(1, frames_to_skip)))
            
            while True:
                if self._is_stopped:
                    # 중간 중단이 발생할 경우, 아직 처리되지 않은 남은 버퍼 강제 처리
                    logic_processor.process_buffer(raw_buffer, force_flush=True)
                    break

                frame_idx += 1
                time_sec = frame_idx / fps

                # skip할 프레임은 grab()으로 디코딩 없이 건너뛰기
                if frame_idx % frames_to_skip != 0:
                    grabbed = cap.grab()
                    if not grabbed: break

                    if frame_idx % fps_int == 0:
                        prog = int((frame_idx / total_frames) * 100)
                        self.progress_signal.emit(prog)
                    continue

                # 처리할 프레임만 실제로 디코딩
                ret, frame = cap.read()
                if not ret: break

                if frame_idx % fps_int == 0:
                    prog = int((frame_idx / total_frames) * 100)
                    self.progress_signal.emit(prog)
                    
                # 1. 감지부: CLIP 이벤트 확정 (1초 간격 상시)
                clip_event = None
                if frame_idx % fps_int == 0 and self.roi_event is not None:
                    clip_event = self._classify_event(frame, device, easy_reader)

                # 2. 감지부: absdiff 기반 변화 감지 → 조건부 OCR (v44 속도 기법)
                # ── Balance ROI 변화 감지 ──
                bx, by, bw, bh = self.roi_bal
                bal_gray = cv2.cvtColor(frame[by:by+bh, bx:bx+bw], cv2.COLOR_BGR2GRAY)
                raw_bal = None

                if prev_bal_gray is not None:
                    if self.stability_pct <= 0:
                        # Stab% = 0 → 변화 감지 OFF, 매 프레임 OCR
                        ocr_result = read_roi(self.roi_bal, frame)
                        if ocr_result is not None:
                            last_ocr_bal = ocr_result
                        raw_bal = last_ocr_bal
                    else:
                        diff_pixels = np.sum(cv2.absdiff(prev_bal_gray, bal_gray) > 25)
                        total_pixels = bw * bh
                        if total_pixels > 0 and diff_pixels < total_pixels * (self.stability_pct / 100.0):  # Stab% 미만 변화 → 안정
                            bal_stable_count += 1
                        else:
                            bal_stable_count = 0  # 변화 발생 → 리셋

                        if bal_stable_count == stability_threshold:
                            # 🎯 안정 확정! 이 순간에만 OCR 1회 호출
                            ocr_result = read_roi(self.roi_bal, frame)
                            if ocr_result is not None:
                                last_ocr_bal = ocr_result
                            raw_bal = last_ocr_bal
                        elif bal_stable_count > stability_threshold:
                            # 안정 유지 중 → 마지막 OCR 값 재사용 (OCR 재호출 안 함)
                            raw_bal = last_ocr_bal
                        # else: bal_stable_count < threshold → 아직 안정되지 않음 → raw_bal = None
                prev_bal_gray = bal_gray.copy()

                # ── Win ROI 변화 감지 ──
                raw_win = 0.0
                if self.roi_win is not None:
                    wx, wy, ww, wh = self.roi_win
                    win_gray = cv2.cvtColor(frame[wy:wy+wh, wx:wx+ww], cv2.COLOR_BGR2GRAY)

                    if prev_win_gray is not None:
                        if self.stability_pct <= 0:
                            # Stab% = 0 → 변화 감지 OFF, 매 프레임 OCR
                            ocr_win_result = read_roi_win(self.roi_win, frame)
                            if ocr_win_result is not None:
                                last_ocr_win = ocr_win_result
                            raw_win = last_ocr_win
                        else:
                            diff_pixels_w = np.sum(cv2.absdiff(prev_win_gray, win_gray) > 25)
                            total_pixels_w = ww * wh
                            if total_pixels_w > 0 and diff_pixels_w < total_pixels_w * (self.stability_pct / 100.0):
                                win_stable_count += 1
                            else:
                                win_stable_count = 0

                            if win_stable_count == stability_threshold:
                                ocr_win_result = read_roi_win(self.roi_win, frame)
                                if ocr_win_result is not None:
                                    last_ocr_win = ocr_win_result
                                raw_win = last_ocr_win
                            elif win_stable_count > stability_threshold:
                                raw_win = last_ocr_win
                        # else: 안정되지 않음 → raw_win = 0.0 유지
                    prev_win_gray = win_gray.copy()

                # 실시간 Raw Data 로그 송출 (UI 연동용)
                hh = int(time_sec // 3600)
                mm = int((time_sec % 3600) // 60)
                ss = int(time_sec % 60)
                
                # 값 포맷팅 방어 로직 (None일 수 있음)
                bal_str = f"{float(raw_bal):,.2f}" if raw_bal is not None else "0.00"
                win_str = f"{float(raw_win):,.2f}" if raw_win is not None else "0.00"
                evt_str = clip_event if clip_event is not None else ""
                
                log_msg = f"[{hh:02d}:{mm:02d}:{ss:02d}] BALANCE: {bal_str} | WIN: {win_str} | EVENT: {evt_str}"
                self.raw_log_signal.emit(log_msg)

                # 3. 버퍼에 쌓기
                raw_buffer.append({
                    "frame_idx": frame_idx,
                    "time_sec": time_sec,
                    "raw_bal": raw_bal,
                    "raw_win": raw_win,
                    "clip_event": clip_event
                })

                # 4. 버퍼 누적 시, 2초 이전 과거의 데이터(미래 문맥 확보분)만 로직 처리기로 밀어내기
                #    예: 현재 30초면, 28초까지의 버퍼만 처리. 미래의 2초(28~30초)는 분석을 위해 홀드
                if len(raw_buffer) > 0:
                    oldest_time = raw_buffer[0]["time_sec"]
                    if time_sec - oldest_time >= 2.0:
                        # 2.0초 시점 이상 차이나는 오래된 데이터들 분류
                        ready_frames = [f for f in raw_buffer if time_sec - f["time_sec"] >= 2.0]
                        if ready_frames:
                            # 로직 처리기에 넘기기
                            logic_processor.process_buffer(ready_frames, force_flush=False)
                            # 남아있어야 할 2초 윈도우 데이터만 유지
                            raw_buffer = [f for f in raw_buffer if time_sec - f["time_sec"] < 2.0]

            # 영상 처리가 완전히 끝나면(EOF), 잔여 버퍼 2초 분량을 강제로 처리
            if not self._is_stopped and raw_buffer:
                logic_processor.process_buffer(raw_buffer, force_flush=True)

            cap.release()
            elapsed = time.time() - start_time
            eh = int(elapsed // 3600)
            em = int((elapsed % 3600) // 60)
            es = int(elapsed % 60)
            self.elapsed_signal.emit(f"{eh:02d}:{em:02d}:{es:02d}")
            self.finished_signal.emit()

        except Exception as e:
            self.error_signal.emit(f"An unexpected error occurred: {str(e)}\n\nDetails: {traceback.format_exc()}")

class LogicProcessor:
    """
    극단순화 스핀 감지 로직:
    - 잔액 최고점(current_bal)은 필터를 통과한 모든 유효값으로 즉시 갱신 (롤업 중 정점 포착)
    - 스핀 확정 조건 1 (하락): '동일한 새로운 잔액'이 3프레임 연속 유지 + 최고점 대비 하락
    - 스핀 확정 조건 2 (상승 안착): 잔액이 '마지막 스핀 확정 잔액(spin_base_bal)'보다
      높은 새 값으로 10프레임 연속 정지 → 롤업 스킵
    - fixed_bet이 설정되면 모든 스핀의 Bet을 해당 값으로 강제 기입
    """
    def __init__(self, fps_int, data_signal, bal_filter=None, fixed_bet=None, drop_only_spin=False):
        self.fps = fps_int
        self.data_signal = data_signal
        self.bal_filter = bal_filter
        self.fixed_bet = fixed_bet
        self.drop_only_spin = drop_only_spin  # Drop Only Spin 모드
        
        self.spin_count = 0
        self.current_bal = None    # 매 프레임 최고점을 추적 (롤업 중에도 즉시 갱신)
        self.current_win = None
        self.last_bet = 0.0
        
        # ★ 핵심: 스핀이 확정된 시점의 잔액 (이 값은 _emit_spin에서만 갱신)
        # 조건 2(상승 안착)는 current_bal이 아니라 이 값과 비교하여
        # 롤업 중간값이 current_bal을 올려버려도 비교 기준이 오염되지 않음
        self.spin_base_bal = None
        
        # 연속 값 확인을 위한 추적기 (하락 감지)
        self.last_raw_bal = None
        self.raw_bal_count = 0
        self.last_stable_bal = None
        
        # 10프레임 상승 안착 카운터 (롤업 스킵 감지)
        self.rise_stable_val = None
        self.rise_stable_count = 0

        # 이벤트 중복 방지: 이전 이벤트와 마지막 감지 시각 추적
        self.last_event = None
        self.last_event_time = None

    def process_buffer(self, frames, force_flush=False):
        for frame_data in frames:
            time_sec = frame_data["time_sec"]
            raw_bal = frame_data["raw_bal"]
            raw_win = frame_data["raw_win"]
            clip_event = frame_data["clip_event"]
            frame_idx = frame_data["frame_idx"]

            # 독립 이벤트 (CLIP 시간용 빈 줄)
            if clip_event:
                # 중복 방지: 이전 이벤트와 동일하면 무시
                should_emit = True
                if self.last_event == clip_event:
                    should_emit = False
                
                # 3초 이상 미감지 시 리셋 (다시 같은 이벤트 감지 허용)
                if self.last_event_time is not None and (time_sec - self.last_event_time) >= 3.0:
                    self.last_event = None
                    should_emit = True
                
                if should_emit:
                    h_m_s = f"{int(time_sec//3600):02d}:{int((time_sec%3600)//60):02d}:{int(time_sec%60):02d}"
                    out_bal = self.current_bal if self.current_bal is not None else 0.0
                    self.data_signal.emit(-1, h_m_s, 0.0, 0.0, float(out_bal), clip_event)
                    self.last_event = clip_event
                
                self.last_event_time = time_sec

            # 1. 이상값 필터 (bal_filter)
            if raw_bal is not None and self.current_bal is not None and self.bal_filter is not None:
                if abs(raw_bal - self.current_bal) >= self.bal_filter:
                    raw_bal = None

            # 필터에 걸러졌거나 OCR이 실패한 경우 즉시 카운터 초기화
            if raw_bal is None:
                self.last_raw_bal = None
                self.raw_bal_count = 0
                self.rise_stable_val = None
                self.rise_stable_count = 0
                continue

            f_bal = float(raw_bal)
            f_win = float(raw_win) if raw_win is not None else 0.0

            # 2. 잔액 최고점(Peak) 추적 (롤업 중 1프레임 정점도 포착)
            if self.current_bal is None or f_bal > self.current_bal:
                self.current_bal = f_bal
                
            # Win 역시 가장 높았던 값을 지속 추적
            if self.current_win is None or f_win > self.current_win:
                self.current_win = f_win

            # 3. 연속 동일 잔액 유지 확인
            if f_bal == self.last_raw_bal:
                self.raw_bal_count += 1
            else:
                self.last_raw_bal = f_bal
                self.raw_bal_count = 1

            # 초기화: 첫 3프레임 안정값을 기준점으로 설정
            if self.spin_base_bal is None:
                if self.raw_bal_count >= 3:
                    self.spin_base_bal = f_bal
                    self.current_bal = f_bal
                    self.current_win = f_win
                    self.last_stable_bal = f_bal
                continue

            # ──────────────────────────────────────────────────
            # 조건 1: N프레임 연속 동일 → 하락(스핀) 여부 검사
            # drop_only_spin ON: 10프레임 + 하락만 스핀 카운트
            # drop_only_spin OFF: 3프레임 (기존)
            # ──────────────────────────────────────────────────
            drop_stable_frames = 3
            
            if self.raw_bal_count == drop_stable_frames:
                stable_bal = f_bal
                
                # 가짜 피크(Noise) 제거
                if self.last_stable_bal is not None and stable_bal == self.last_stable_bal:
                    if self.current_bal > stable_bal:
                        self.current_bal = stable_bal
                else:
                    # 하락 확인 (최고점 대비)
                    bal_diff = self.current_bal - stable_bal
                    
                    if bal_diff >= 0.01:
                        # 🎈 하락 감지 확정! 스핀 1회
                        self._emit_spin(time_sec, bal_diff, stable_bal)
                        
                        # 상승 안착 카운터도 리셋
                        self.rise_stable_val = None
                        self.rise_stable_count = 0
                        
                    self.last_stable_bal = stable_bal

            # ──────────────────────────────────────────────────
            # 조건 2: 10프레임 연속 동일 + 상승 → 롤업 안착 처리
            # drop_only_spin OFF: 스핀 카운트 (기존 동작)
            # drop_only_spin ON:  스핀 카운트 안 함, baseline만 갱신
            # ──────────────────────────────────────────────────
            if self.spin_base_bal is not None and f_bal > self.spin_base_bal:
                # spin_base_bal보다 높은 값이므로 상승 안착 후보
                if f_bal == self.rise_stable_val:
                    self.rise_stable_count += 1
                else:
                    self.rise_stable_val = f_bal
                    self.rise_stable_count = 1
                
                if self.rise_stable_count >= 10:
                    if self.drop_only_spin:
                        # 🔇 Drop Only: baseline만 조용히 갱신 (스핀 카운트 안 함)
                        self.current_bal = f_bal
                        self.current_win = 0.0
                        self.last_stable_bal = f_bal
                        self.spin_base_bal = f_bal
                    else:
                        # 🎈 상승 안착 확정! 롤업 스킵 스핀 (기존 동작)
                        bal_diff = f_bal - self.spin_base_bal
                        self._emit_spin(time_sec, bal_diff, f_bal)
                    
                    self.rise_stable_val = None
                    self.rise_stable_count = 0
            else:
                # spin_base_bal 이하이면 상승 안착 카운터 리셋
                self.rise_stable_val = None
                self.rise_stable_count = 0

        pass

    def _emit_spin(self, time_sec, bal_diff, final_bal):
        """스핀 1회 확정 시 데이터 방출"""
        self.spin_count += 1
        
        # Bet 결정: fixed_bet이 있으면 무조건 사용, 없으면 계산된 차액 사용
        if self.fixed_bet is not None:
            bet_amount = self.fixed_bet
        else:
            bet_amount = round(bal_diff, 2)
        self.last_bet = bet_amount
        
        h_m_s = f"{int(time_sec//3600):02d}:{int((time_sec%3600)//60):02d}:{int(time_sec%60):02d}"
        win_amount = self.current_win if self.current_win is not None else 0.0
        
        self.data_signal.emit(self.spin_count, h_m_s, self.last_bet, win_amount, final_bal, "")
        
        # 다음 스핀 추적을 위해 리셋
        self.current_bal = final_bal
        self.current_win = 0.0
        self.last_stable_bal = final_bal
        self.spin_base_bal = final_bal  # ★ 스핀 확정 잔액 갱신
