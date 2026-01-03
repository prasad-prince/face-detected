import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import pygame
import time
from collections import deque
from ultralytics import YOLO
import os

class StudyMonitor:
    def __init__(self):
        # Initialize YOLO for phone detection
        print("Loading YOLO model...")
        self.yolo_model = YOLO('yolov8n.pt')  # Using nano model for speed
        
        # Initialize OpenCV Face Detection
        print("Loading Face Detection model...")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize audio
        pygame.mixer.init()
        self.create_audio_alert()
        
        # Tracking variables
        self.phone_detected = False
        self.face_detected = False
        self.last_alert_time = 0
        self.alert_cooldown = 5  # seconds between alerts
        
        # Distraction logging
        self.distraction_log = []
        self.phone_detection_start = None
        self.total_distraction_time = 0
        
        # Statistics
        self.frame_count = 0
        self.phone_detections = 0
        self.no_face_count = 0
        
        # Display settings
        self.warning_display_time = 0
        
    def create_audio_alert(self):
        """Create a simple beep sound as alert or load custom audio file"""
        if os.path.exists('alert.wav'):
            self.alert_sound = pygame.mixer.Sound('alert.wav')
            print("✓ Loaded custom alert sound: alert.wav")
        else:
            # Generate a simple beep sound
            sample_rate = 22050
            duration = 0.5
            frequency = 800
            
            # Generate sine wave
            t = np.linspace(0, duration, int(sample_rate * duration))
            wave = np.sin(2 * np.pi * frequency * t)
            
            # Convert to 16-bit format
            wave = (wave * 32767).astype(np.int16)
            
            # Create stereo sound
            stereo_wave = np.column_stack((wave, wave))
            
            self.alert_sound = pygame.sndarray.make_sound(stereo_wave)
            print("✓ Using generated beep sound (add alert.wav for custom sound)")
    
    def play_alert(self):
        """Play audio alert with cooldown"""
        current_time = time.time()
        if current_time - self.last_alert_time > self.alert_cooldown:
            self.alert_sound.play()
            self.last_alert_time = current_time
            self.warning_display_time = current_time
            print(f"⚠️  ALERT: Phone detected at {datetime.now().strftime('%H:%M:%S')}")
    
    def detect_phone(self, frame):
        """Detect mobile phone using YOLO"""
        results = self.yolo_model(frame, verbose=False)
        phone_found = False
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Class 67 is 'cell phone' in COCO dataset
                if cls == 67 and conf > 0.5:
                    phone_found = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, f'Phone {conf:.2f}', (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return phone_found, frame
    
    def detect_face(self, frame):
        """Detect face using OpenCV"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # More sensitive face detection
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05,  # Smaller scale factor for more detection
            minNeighbors=3,    # Lower threshold for more detections
            minSize=(30, 30)   # Minimum face size
        )
        face_found = len(faces) > 0
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, 'Face Detected', (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return face_found, frame
    
    def log_distraction(self, start_time, end_time):
        """Log distraction event"""
        duration = end_time - start_time
        self.distraction_log.append({
            'start': datetime.fromtimestamp(start_time).strftime('%H:%M:%S'),
            'end': datetime.fromtimestamp(end_time).strftime('%H:%M:%S'),
            'duration': duration
        })
        self.total_distraction_time += duration
    
    def draw_interface(self, frame):
        """Draw monitoring interface"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay for stats
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Title
        cv2.putText(frame, 'AI Study Monitor', (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Status indicators
        face_status = "✓ Detected" if self.face_detected else "✗ Not Detected"
        face_color = (0, 255, 0) if self.face_detected else (0, 0, 255)
        cv2.putText(frame, f'Face: {face_status}', (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)
        
        phone_status = "✗ Clear" if not self.phone_detected else "⚠ DETECTED!"
        phone_color = (0, 255, 0) if not self.phone_detected else (0, 0, 255)
        cv2.putText(frame, f'Phone: {phone_status}', (20, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, phone_color, 2)
        
        # Statistics
        cv2.putText(frame, f'Distractions: {len(self.distraction_log)}', (20, 135),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f'Total Time: {int(self.total_distraction_time)}s', (20, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Warning message when phone detected
        if self.phone_detected and time.time() - self.warning_display_time < 3:
            warning = "PUT AWAY YOUR PHONE!"
            text_size = cv2.getTextSize(warning, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h - 50
            
            # Warning background
            cv2.rectangle(frame, (text_x-10, text_y-text_size[1]-10),
                         (text_x+text_size[0]+10, text_y+10), (0, 0, 255), -1)
            cv2.putText(frame, warning, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Instructions
        cv2.putText(frame, 'Press Q to quit | S to save report', (10, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def save_report(self):
        """Save distraction report to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'study_report_{timestamp}.txt'
        
        with open(filename, 'w') as f:
            f.write("="*50 + "\n")
            f.write("AI STUDY MONITOR - SESSION REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"Total Distractions: {len(self.distraction_log)}\n")
            f.write(f"Total Distraction Time: {int(self.total_distraction_time)} seconds\n")
            f.write(f"Average Distraction: {int(self.total_distraction_time/len(self.distraction_log)) if self.distraction_log else 0} seconds\n\n")
            
            if self.distraction_log:
                f.write("Distraction Log:\n")
                f.write("-"*50 + "\n")
                for i, log in enumerate(self.distraction_log, 1):
                    f.write(f"{i}. Start: {log['start']} | End: {log['end']} | Duration: {log['duration']:.1f}s\n")
            
            f.write("\n" + "="*50 + "\n")
        
        print(f"✓ Report saved: {filename}")
        return filename
    
    def run(self, demo_mode=False):
        """Main monitoring loop"""
        print("\n" + "="*50)
        print("AI STUDY MONITORING SYSTEM")
        print("="*50)
        
        if demo_mode:
            print("DEMO MODE: Running without webcam")
            print("✓ Using simulated data for demonstration")
            self.run_demo()
            return
        
        print("Starting webcam...")
        
        # Try to access webcam, try different indices if 0 doesn't work
        cap = None
        for i in range(5):  # Try indices 0 to 4
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"✓ Webcam accessed successfully on index {i}")
                break
        else:
            print("ERROR: Cannot access webcam on any index (0-4)!")
            print("Please check:")
            print("- Webcam is not in use by another application")
            print("- Webcam permissions are granted")
            print("- Webcam is properly connected")
            print("\nAlternatively, run with demo mode: python 'ai project.py' demo")
            return
        
        print("✓ Webcam started successfully")
        print("\nMonitoring active...")
        print("- Phone detection: ENABLED")
        print("- Face tracking: ENABLED")
        print("- Audio alerts: ENABLED")
        print("\nPress 'Q' to quit | Press 'S' to save report\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("ERROR: Failed to capture frame")
                    break
                
                self.frame_count += 1
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect face
                self.face_detected, frame = self.detect_face(frame)
                
                # Detect phone
                phone_detected_now, frame = self.detect_phone(frame)
                
                # Handle phone detection state changes
                if phone_detected_now and not self.phone_detected:
                    # Phone just appeared
                    self.phone_detected = True
                    self.phone_detection_start = time.time()
                    self.play_alert()
                    self.phone_detections += 1
                    
                elif not phone_detected_now and self.phone_detected:
                    # Phone removed
                    self.phone_detected = False
                    if self.phone_detection_start:
                        self.log_distraction(self.phone_detection_start, time.time())
                        self.phone_detection_start = None
                
                # Track no face
                if not self.face_detected:
                    self.no_face_count += 1
                
                # Draw interface
                frame = self.draw_interface(frame)
                
                # Display frame
                cv2.imshow('AI Study Monitor', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\n\nShutting down...")
                    break
                elif key == ord('s') or key == ord('S'):
                    self.save_report()
        
        finally:
            # Cleanup
            if self.phone_detected and self.phone_detection_start:
                self.log_distraction(self.phone_detection_start, time.time())
            
            cap.release()
            cv2.destroyAllWindows()
            
            # Final report
            print("\n" + "="*50)
            print("SESSION SUMMARY")
            print("="*50)
            print(f"Total distractions: {len(self.distraction_log)}")
            print(f"Total distraction time: {int(self.total_distraction_time)} seconds")
            print(f"Frames processed: {self.frame_count}")
            
            # Auto-save report
            filename = self.save_report()
            print(f"\nFull report saved to: {filename}")
            print("\nThank you for using AI Study Monitor!")

    def run_demo(self):
        """Run in demo mode without webcam - simulates monitoring"""
        print("\nDemo mode active...")
        print("- Phone detection: SIMULATED")
        print("- Face tracking: SIMULATED")
        print("- Audio alerts: ENABLED")
        print("\nPress 'Q' to quit | Press 'S' to save report | Press 'P' to simulate phone detection\n")
        
        # Create a demo window
        demo_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        try:
            while True:
                self.frame_count += 1
                
                # Simulate random face detection
                self.face_detected = np.random.random() > 0.1  # 90% chance of face detected
                
                # Simulate occasional phone detection
                phone_detected_now = np.random.random() > 0.95  # 5% chance per frame
                
                # Handle phone detection state changes
                if phone_detected_now and not self.phone_detected:
                    self.phone_detected = True
                    self.phone_detection_start = time.time()
                    self.play_alert()
                    self.phone_detections += 1
                    print("DEMO: Phone detected!")
                    
                elif not phone_detected_now and self.phone_detected:
                    self.phone_detected = False
                    if self.phone_detection_start:
                        self.log_distraction(self.phone_detection_start, time.time())
                        self.phone_detection_start = None
                
                # Track no face
                if not self.face_detected:
                    self.no_face_count += 1
                
                # Draw interface on demo frame
                demo_frame = self.draw_interface(demo_frame)
                
                # Add demo text
                cv2.putText(demo_frame, 'DEMO MODE - No Camera Required', (200, 250),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Display demo frame
                cv2.imshow('AI Study Monitor - DEMO MODE', demo_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(100) & 0xFF  # Slower update for demo
                if key == ord('q') or key == ord('Q'):
                    print("\n\nShutting down demo...")
                    break
                elif key == ord('s') or key == ord('S'):
                    self.save_report()
                elif key == ord('p') or key == ord('P'):
                    # Manual phone trigger for demo
                    if not self.phone_detected:
                        self.phone_detected = True
                        self.phone_detection_start = time.time()
                        self.play_alert()
                        self.phone_detections += 1
                        print("DEMO: Manual phone detection triggered!")
        
        finally:
            cv2.destroyAllWindows()
            
            # Final report
            print("\n" + "="*50)
            print("DEMO SESSION SUMMARY")
            print("="*50)
            print(f"Total distractions: {len(self.distraction_log)}")
            print(f"Total distraction time: {int(self.total_distraction_time)} seconds")
            print(f"Frames processed: {self.frame_count}")
            
            # Auto-save report
            filename = self.save_report()
            print(f"\nDemo report saved to: {filename}")
            print("\nThank you for trying AI Study Monitor Demo!")

if __name__ == "__main__":
    import sys
    demo_mode = len(sys.argv) > 1 and sys.argv[1].lower() == 'demo'
    
    monitor = StudyMonitor()
    monitor.run(demo_mode=demo_mode)