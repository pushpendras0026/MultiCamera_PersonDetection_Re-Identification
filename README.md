# Multi-Camera Person Detection & Re-Identification

This repository contains two distinct subsystems for person tracking and re-identification:
1. **Multi-Camera Person Tracking & Re-Identification** (Real-time tracking across multiple cameras)
2. **Video-based Re-Identification (STMN)** (Spatial-Temporal Memory Networks for offline video-to-video matching)

---

## Part 1: Multi-Camera Tracking (Real-Time Dual Camera Setup)

This module performs real-time YOLOv4 detection and DeepSORT tracking across multiple video feeds, allowing you to use dual mobile cameras via DroidCam over a local network.

### Step-by-Step Setup

1. **Install Dependencies**:
   Ensure you have activated your Python virtual environment, then install the required packages:
   ```bash
   pip install -r Multi-Camera-Person-Tracking-and-Re-Identification/requirements.txt
   ```
2. **Setup DroidCam on Mobile Devices**:
   - Install the **DroidCam** app on two smartphones.
   - Connect both smartphones and your PC to the **same Wi-Fi network**.
   - Open DroidCam on both phones and note down the `IP address` and `Port` (e.g., `192.168.1.10:4747`).

3. **Running the Tracker with Dual Mobile Cameras**:
   Navigate to the tracking folder:
   ```bash
   cd Multi-Camera-Person-Tracking-and-Re-Identification
   ```
   You can run the tracking script by feeding it the HTTP video streams from both DroidCam instances:
   ```bash
   python demo.py --webcam "http://<PHONE_1_IP>:<PORT>/video" "http://<PHONE_2_IP>:<PORT>/video" --version v4
   ```
   *Example:*
   ```bash
   python demo.py --webcam http://192.168.1.10:4747/video http://192.168.1.11:4747/video --version v4
   ```
   
   *(Note: If you have configured DroidCam Client on Windows to map your phones to local webcams, you can simply use `--webcam 0 1`)*.

---

## Part 2: Video-Based Re-Identification (STMN)

This module is an implementation of "Video-based Person Re-identification with Spatial and Temporal Memory Networks" (arXiv:2108.09039), designed to match tracked sequences offline by removing spatial distractors and aggregating temporal context.

### Step-by-Step Setup

1. **Install Dependencies**:
   Navigate to the STMN directory and ensure libraries are installed:
   ```bash
   cd STMN_ReID
   pip install -r requirements.txt
   ```
   *(This uses PyTorch with CUDA for GPU acceleration).*

2. **Prepare Datasets**:
   - Download the target datasets (MARS, iLIDS-VID) and place them inside the `STMN_ReID/database/` directory.

3. **Training the Model**:
   We have optimized batch files to run the training pipelines with Automatic Mixed Precision (AMP) for RTX 4060 (8GB VRAM).
   - To train on **MARS**:
     ```bash
     cd smem_tmem
     train_mars.bat
     ```
   - To train on **iLIDS-VID**:
     ```bash
     cd smem_tmem
     train_ilids.bat
     ```

4. **Evaluating the Model**:
   Once training completes, model checkpoints are saved in the `checkpoints/` directory. Evaluate the metrics (Rank-1, Rank-5, mAP) using:
   - For **MARS**: `evaluate_mars.bat`
   - For **iLIDS-VID**: `evaluate_ilids.bat`

## Additional Documents
Please check the `presentation` and PDF reports in the root directory for technical methodology, ablation studies, and qualitative results.
