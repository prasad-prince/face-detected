"""
AI Study Monitor - College Mini Project
=======================================

This project implements a real-time study monitoring system using computer vision.
It detects student faces and mobile phones to monitor study sessions and alert
when distractions occur.

Features:
- Face detection using OpenCV Haar Cascades
- Phone detection using YOLOv8
- Audio alerts when phone detected while face is visible
- Real-time monitoring interface
- Distraction logging and reporting
- Demo mode for testing without webcam

Author: [Your Name]
Date: January 2026
"""

import cv2
import numpy as np
from datetime import datetime
import pygame
import time
import os
import sys
from ultralytics import YOLO

class WebcamHandler:
    """Handles webcam initialization and frame capture"""
    
    def __init__(self):
        self.cap = None
        self.camera_index = -1
        
    def initialize_camera(self):
        """Try to access webcam on different indices"""
        for i in range(5):  # Try indices 0-4
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.cap = cap
                self.camera_index = i
                print(f"✓ Webcam accessed successfully on index {i}")
                return True
        return False
    
    def read_frame(self):
        """Read a frame from the webcam"""
        if self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)  # Mirror effect
        return ret, frame
    
    def release(self):
        """Release the webcam"""
        if self.cap:
            self.cap.release()

class FaceDetector:
    """Handles face detection using OpenCV Haar Cascades"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if self.face_cascade.empty():
            raise ValueError("Error: Could not load face cascade classifier")
    
    def detect(self, frame):
        """Detect faces in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Optimized parameters for better detection
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05,  # Smaller for more detection
            minNeighbors=3,    # Lower for more sensitivity
            minSize=(30, 30)   # Minimum face size
        )
        
        # Draw rectangles and labels for detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, 'Face Detected', (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return len(faces) > 0, frame

class PhoneDetector:
    """Handles mobile phone detection using YOLOv8"""
    
    def __init__(self):
        try:
            self.model = YOLO('yolov8n.pt')  # Load YOLOv8 nano model
            print("✓ YOLO model loaded successfully")
        except Exception as e:
            raise ValueError(f"Error loading YOLO model: {e}")
    
    def detect(self, frame):
        """Detect phones in the frame"""
        phone_found = False
        
        try:
            results = self.model(frame, verbose=False, conf=0.5)  # Confidence threshold
            
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
                        cv2.putText(frame, 'Phone Detected - Focus on Study', (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        except Exception as e:
            print(f"Warning: Phone detection error: {e}")
        
        return phone_found, frame

class AudioAlert:
    """Handles audio alert system"""
    
    def __init__(self):
        pygame.mixer.init()
        self.alert_sound = None
        self.last_alert_time = 0
        self.cooldown = 5  # seconds between alerts
        
        # Try to load custom alert sound
        if os.path.exists('alert.wav'):
            try:
                self.alert_sound = pygame.mixer.Sound('alert.wav')
                print("✓ Loaded custom alert sound: alert.wav")
            except Exception as e:
                print(f"Warning: Could not load custom sound: {e}")
                self._create_default_sound()
        else:
            self._create_default_sound()
    
    def _create_default_sound(self):
        """Create a default beep sound"""
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
        print("✓ Using default beep sound (add alert.wav for custom sound)")
    
    def play_alert(self, face_detected=False):
        """
        Play alert only if face is detected and cooldown has passed
        This ensures alert only when student is present but distracted
        """
        current_time = time.time()
        
        if face_detected and (current_time - self.last_alert_time) > self.cooldown:
            try:
                self.alert_sound.play()
                self.last_alert_time = current_time
                print(f"⚠️ ALERT: Phone detected while face visible at {datetime.now().strftime('%H:%M:%S')}")
                return True
            except Exception as e:
                print(f"Warning: Could not play alert sound: {e}")
        
        return False

class DistractionLogger:
    """Handles logging of distraction events"""
    
    def __init__(self):
        self.distraction_log = []
        self.phone_detection_start = None
        self.total_distraction_time = 0
    
    def start_distraction(self, timestamp):
        """Mark start of distraction period"""
        if self.phone_detection_start is None:
            self.phone_detection_start = timestamp
    
    def end_distraction(self, timestamp):
        """Mark end of distraction period and log it"""
        if self.phone_detection_start:
            duration = timestamp - self.phone_detection_start
            self.distraction_log.append({
                'start': datetime.fromtimestamp(self.phone_detection_start).strftime('%H:%M:%S'),
                'end': datetime.fromtimestamp(timestamp).strftime('%H:%M:%S'),
                'duration': duration
            })
            self.total_distraction_time += duration
            self.phone_detection_start = None
    
    def save_report(self):
        """Save distraction report to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'study_report_{timestamp}.txt'
        
        try:
            with open(filename, 'w') as f:
                f.write("="*60 + "\n")
                f.write("AI STUDY MONITOR - SESSION REPORT\n")
                f.write("="*60 + "\n\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write(f"Total Distractions: {len(self.distraction_log)}\n")
                f.write(f"Total Distraction Time: {int(self.total_distraction_time)} seconds\n")
                
                if self.distraction_log:
                    avg_time = self.total_distraction_time / len(self.distraction_log)
                    f.write(f"Average Distraction Duration: {avg_time:.1f} seconds\n\n")
                    
                    f.write("Distraction Details:\n")
                    f.write("-"*60 + "\n")
                    for i, log in enumerate(self.distraction_log, 1):
                        f.write(f"{i}. Start: {log['start']} | End: {log['end']} | Duration: {log['duration']:.1f}s\n")
                else:
                    f.write("No distractions recorded - Great focus!\n")
                
                f.write("\n" + "="*60 + "\n")
            
            print(f"✓ Report saved: {filename}")
            return filename
        
        except Exception as e:
            print(f"Error saving report: {e}")
            return None

class StudyMonitorUI:
    """Handles the monitoring interface display"""
    
    def __init__(self):
        self.warning_display_time = 0
    
    def draw_interface(self, frame, face_detected, phone_detected, distraction_logger):
        """Draw the monitoring interface overlay"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay for stats
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Title
        cv2.putText(frame, 'AI Study Monitor', (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Status indicators
        face_status = "✓ Detected" if face_detected else "✗ Not Detected"
        face_color = (0, 255, 0) if face_detected else (0, 0, 255)
        cv2.putText(frame, f'Face: {face_status}', (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)
        
        phone_status = "✗ Clear" if not phone_detected else "⚠ DETECTED!"
        phone_color = (0, 255, 0) if not phone_detected else (0, 0, 255)
        cv2.putText(frame, f'Phone: {phone_status}', (20, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, phone_color, 2)
        
        # Statistics
        cv2.putText(frame, f'Distractions: {len(distraction_logger.distraction_log)}', (20, 135),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f'Total Time: {int(distraction_logger.total_distraction_time)}s', (20, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Warning message when phone detected
        if phone_detected and time.time() - self.warning_display_time < 3:
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

class StudyMonitor:
    """
    Main Study Monitor class that coordinates all components
    
    This class implements a computer vision system for monitoring study sessions.
    It uses separate components for webcam handling, face detection, phone detection,
    audio alerts, and logging to maintain clean separation of concerns.
    """
    
    def __init__(self):
        """Initialize all components with error handling"""
        try:
            print("Initializing AI Study Monitor...")
            
            # Initialize components
            self.webcam = WebcamHandler()
            self.face_detector = FaceDetector()
            self.phone_detector = PhoneDetector()
            self.audio_alert = AudioAlert()
            self.logger = DistractionLogger()
            self.ui = StudyMonitorUI()
            
            # Tracking variables
            self.phone_detected = False
            self.frame_count = 0
            
            print("✓ All components initialized successfully")
            
        except Exception as e:
            print(f"❌ Initialization error: {e}")
            raise
    
    def run_monitoring(self):
        """Main monitoring loop"""
        print("\n" + "="*60)
        print("AI STUDY MONITORING SYSTEM")
        print("="*60)
        
        # Initialize webcam
        if not self.webcam.initialize_camera():
            print("❌ ERROR: Cannot access webcam!")
            print("Please check:")
            print("- Webcam is connected and not in use")
            print("- Camera permissions are granted")
            return
        
        print("\n🎯 Monitoring active...")
        print("- Face detection: ENABLED")
        print("- Phone detection: ENABLED") 
        print("- Audio alerts: ENABLED (only when face + phone detected)")
        print("\nControls: Q=Quit | S=Save Report\n")
        
        try:
            while True:
                # Read frame
                ret, frame = self.webcam.read_frame()
                if not ret:
                    print("❌ ERROR: Failed to capture frame")
                    break
                
                self.frame_count += 1
                
                # Detect face and phone
                face_detected, frame = self.face_detector.detect(frame)
                phone_detected_now, frame = self.phone_detector.detect(frame)
                
                # Handle phone detection state changes
                if phone_detected_now and not self.phone_detected:
                    # Phone just appeared
                    self.phone_detected = True
                    self.logger.start_distraction(time.time())
                    
                    # Play alert ONLY if face is also detected
                    if face_detected:
                        self.audio_alert.play_alert(face_detected=True)
                        self.ui.warning_display_time = time.time()
                    
                elif not phone_detected_now and self.phone_detected:
                    # Phone removed - end distraction
                    self.phone_detected = False
                    self.logger.end_distraction(time.time())
                
                # Draw interface
                frame = self.ui.draw_interface(frame, face_detected, self.phone_detected, self.logger)
                
                # Display frame
                cv2.imshow('AI Study Monitor', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\n⏹️ Shutting down...")
                    break
                elif key == ord('s') or key == ord('S'):
                    self.logger.save_report()
        
        finally:
            # Cleanup
            if self.phone_detected and self.logger.phone_detection_start:
                self.logger.end_distraction(time.time())
            
            self.webcam.release()
            cv2.destroyAllWindows()
            
            # Final report
            print("\n" + "="*60)
            print("SESSION SUMMARY")
            print("="*60)
            print(f"Total frames processed: {self.frame_count}")
            print(f"Total distractions: {len(self.logger.distraction_log)}")
            print(f"Total distraction time: {int(self.logger.total_distraction_time)} seconds")
            
            # Auto-save final report
            filename = self.logger.save_report()
            if filename:
                print(f"\n📄 Final report saved to: {filename}")
            print("\n🙏 Thank you for using AI Study Monitor!")

def main():
    """Main function with demo mode support"""
    demo_mode = len(sys.argv) > 1 and sys.argv[1].lower() == 'demo'
    
    if demo_mode:
        print("Demo mode not implemented in refactored version")
        print("Please run without 'demo' argument for full functionality")
        return
    
    try:
        monitor = StudyMonitor()
        monitor.run_monitoring()
    except KeyboardInterrupt:
        print("\n⏹️ Monitoring stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        print("Please check your setup and try again")

if __name__ == "__main__":
    main()