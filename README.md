# AI Study Monitor - College Mini Project

## 📋 Project Overview

This is a computer vision-based study monitoring system that uses AI to detect student faces and mobile phones in real-time. The system alerts students when they get distracted during study sessions and maintains detailed logs of their focus patterns.

## 🎯 Key Features

- **Real-time Face Detection**: Uses OpenCV Haar Cascades to detect student presence
- **Phone Detection**: YOLOv8 model identifies mobile phones as distractions
- **Smart Audio Alerts**: Only triggers when phone detected while student is present
- **Distraction Logging**: Records timestamps and duration of distractions
- **Interactive UI**: Live monitoring interface with statistics
- **Report Generation**: Automatic text reports with session summaries

## 🏗️ Architecture & Improvements

### **Modular Design**
The code has been completely refactored into separate classes for better maintainability:

- `WebcamHandler`: Manages camera initialization and frame capture
- `FaceDetector`: Handles face detection using OpenCV
- `PhoneDetector`: Manages phone detection with YOLOv8
- `AudioAlert`: Controls audio notifications with cooldown logic
- `DistractionLogger`: Records and reports distraction events
- `StudyMonitorUI`: Manages the visual interface
- `StudyMonitor`: Main coordinator class

### **Key Improvements Made**

1. **Separation of Concerns**: Each component has a single responsibility
2. **Error Handling**: Comprehensive try-catch blocks for camera/model failures
3. **Smart Alert Logic**: Audio alerts only when face + phone detected simultaneously
4. **Cooldown System**: Prevents continuous audio playback (5-second cooldown)
5. **Better Labels**: Clear on-screen text ("Face Detected", "Phone Detected - Focus on Study")
6. **Performance Optimization**: YOLO confidence threshold, optimized Haar parameters
7. **Code Comments**: Extensive documentation for viva preparation
8. **Beginner-Friendly**: Clean, readable code suitable for college projects

### **Technical Specifications**

- **Language**: Python 3.8+
- **Libraries**:
  - OpenCV 4.x (Face detection)
  - Ultralytics YOLOv8 (Phone detection)
  - Pygame (Audio alerts)
  - NumPy (Data processing)
- **Models**: YOLOv8n (6.2MB), OpenCV Haar Cascades (built-in)
- **Platform**: Windows/Linux/Mac

## 🚀 Installation & Setup

### Prerequisites
```bash
pip install opencv-python ultralytics pygame numpy
```

### Download Models
The YOLOv8 model downloads automatically on first run. Haar cascades are included with OpenCV.

## 📖 Usage

### Basic Usage
```bash
python "ai project.py"
```

### Advanced Modes

**Demo Mode** (No webcam required):
```bash
python "ai project.py" demo
```
- Simulates monitoring interface for testing
- Perfect for presentations and development
- Includes all UI elements and simulated detections

**Audio Test Mode** (Test audio system only):
```bash
python "ai project.py" test-audio
```
- Independently tests audio functionality
- Useful for troubleshooting sound issues
- Validates pygame mixer setup

### Controls
- **Q**: Quit the application
- **S**: Save current session report
- **P** (Demo mode only): Manually trigger phone detection

### Custom Audio
Add `alert.wav` file to the project directory for custom alert sounds.

## 🔧 How It Works

### 1. Initialization Phase
- Loads YOLOv8 model for phone detection
- Initializes OpenCV face cascade classifier
- Sets up audio system with pygame
- Initializes webcam on available camera index

### 2. Monitoring Loop
- Captures frames from webcam
- Detects faces using Haar cascades
- Detects phones using YOLOv8
- **Triggers alert only when**: Phone detected AND face is visible
- Updates live statistics and interface
- Logs distraction events with timestamps

### 3. Alert System
- **Condition**: Phone visible + Face visible + Cooldown expired
- **Cooldown**: 5 seconds between alerts
- **Audio**: Custom WAV or generated beep
- **Visual**: On-screen warning message

### 4. Reporting
- Automatic timestamp logging
- Session summary with statistics
- Text file reports with distraction details

## 📊 Sample Output

### Normal Monitoring Mode
```
==================================================
AI STUDY MONITORING SYSTEM
==================================================
✓ Webcam accessed successfully on index 0

🎯 Monitoring active...
- Face detection: ENABLED
- Phone detection: ENABLED
- Audio alerts: ENABLED (only when face + phone detected)

⚠️ ALERT: Phone detected while face visible at 14:30:25

==================================================
SESSION SUMMARY
==================================================
Total frames processed: 1250
Total distractions: 3
Total distraction time: 45 seconds
```

### Demo Mode Output
```
🎮 DEMO MODE - Testing without webcam
This mode simulates the monitoring interface for development/testing

Demo features:
- Simulated face and phone detection
- Working audio alerts
- Report generation
- All UI elements

🎭 DEMO: Phone detected!
🔊 ALERT: Phone detected while face visible
```

### Audio Test Mode Output
```
🔊 AUDIO TEST MODE
Testing audio system independently...
✓ Audio system initialized successfully
✓ Using default beep sound (add alert.wav for custom sound)

🔊 Testing Audio System...
▶️  Playing test sound...
✅ Audio test successful!

✅ Audio system is working correctly!
```

## 🎓 College Project Features

### Viva-Ready Explanations

**Q: How does face detection work?**
- Uses OpenCV's Haar Cascade Classifier trained on thousands of face images
- Converts frame to grayscale for faster processing
- Applies sliding window technique to detect facial features
- Parameters optimized for real-time performance

**Q: Why YOLOv8 for phone detection?**
- State-of-the-art object detection model
- Fast inference (nano model: ~6.2MB)
- Pre-trained on COCO dataset (includes cell phones as class 67)
- Confidence thresholding prevents false positives

**Q: How is distraction logging implemented?**
- Tracks phone detection start/end times
- Calculates duration of each distraction event
- Stores in structured format with timestamps
- Generates human-readable reports

**Q: What prevents false alerts?**
- Face presence verification before alerts
- 5-second cooldown between audio alerts
- Confidence thresholds for detection accuracy
- State machine prevents continuous triggering

### Code Quality
- **Modular**: Each component in separate class
- **Documented**: Extensive comments and docstrings
- **Error Handling**: Graceful failure recovery
- **Performance**: Optimized for real-time processing
- **Maintainable**: Clean separation of concerns

## 🔍 Troubleshooting

### Common Issues

**Camera not found:**
- Check camera permissions in OS settings
- Ensure no other apps are using the camera
- Try different camera indices (0, 1, 2...)

**Poor detection:**
- Ensure good lighting
- Position face centrally in frame
- Clean camera lens

**Audio not working:**
- Check system audio settings
- Add `alert.wav` file for custom sound
- Ensure pygame is properly installed

**Performance issues:**
- Close other applications
- Use lower resolution if needed
- YOLOv8n is optimized for speed

## 📈 Future Enhancements

- Multiple face tracking
- Eye gaze detection for attention monitoring
- Machine learning for distraction prediction
- Web interface for remote monitoring
- Integration with study planning apps

## 📝 License & Credits

This project is developed for educational purposes as a college mini project.

**Libraries Used:**
- OpenCV: Computer vision framework
- YOLOv8: Object detection by Ultralytics
- Pygame: Audio handling
- NumPy: Numerical computing

---

**Happy Studying! 📚✨**