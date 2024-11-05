# AI Virtual Mouse

This project implements an AI-based virtual mouse using hand tracking. It leverages computer vision and machine learning techniques to detect hand gestures and control the mouse pointer accordingly. The project uses OpenCV for image processing, MediaPipe for hand tracking, and PyAutoGUI and Pynput for controlling the mouse and keyboard.

## Features

- Hand detection and tracking using MediaPipe.
- Control mouse pointer using hand movements.
- Click, drag, and scroll using hand gestures.
- Switch between windows using hand gestures.

## Requirements

- Python 3.6+
- OpenCV
- MediaPipe
- NumPy
- PyAutoGUI
- Pynput

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/ai-virtual-mouse.git
    cd ai-virtual-mouse
    ```

2. Install the required packages:
    ```sh
    pip install opencv-python mediapipe numpy pyautogui pynput
    ```
    or

    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the `AiVirtualMouseProject.py` script:
    ```sh
    python AiVirtualMouseProject.py
    ```

2. The script will open your webcam and start detecting your hand. Use the following gestures to control the mouse:
    - **Move Mouse**: Open your Palm towards the camera and move your hand, to move the mouse pointer.
    - **Left Click**: Touch your thumb and index finger together while keeping the other fingers/palm open.
    - **Switch Windows**: Turn your palm away from the camera; a window-switch menu will appear. Keep your palm facing away from the camera and move your hand (mouse pointer) to the desired window; then turn your palm towards the camera to select the window.
    - **Drag and Drop**: Touch your thumb and index finger together and move your hand while keeping the fingers together.
    - **Scroll**: (Not implemented) Use your thumb and index finger to scroll.

## Best Use Cases

- **Presentation Control**: Use hand gestures to control slides during a presentation.
- **Accessibility**: Provide an alternative input method for users with physical disabilities.
- **Touchless Interaction**: Control your computer without touching the mouse or keyboard, useful in hygienic environments.

## How to Replicate

1. Ensure you have a webcam connected to your computer.
2. Follow the installation steps to set up the environment.
3. Run the `AiVirtualMouseProject.py` script.
4. Experiment with different hand gestures to control the mouse and keyboard.

## NOTE
* The webcam should be placed at a comfortable distance and angle to detect hand gestures accurately.
* The hand detection model may not work well in low-light conditions or with complex backgrounds.
* The hand tracking model may not work well with multiple hands or occlusions.

## License

I recommend using the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license for this project. This license allows others to remix, tweak, and build upon your work non-commercially, as long as they credit you and license their new creations under the identical terms.

```markdown
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

You are free to:
- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material

Under the following terms:
- Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- NonCommercial — You may not use the material for commercial purposes.
- ShareAlike — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

For the full license text, please refer to the LICENSE file in the repository.
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Acknowledgements

- [OpenCV](https://opencv.org/)
- [MediaPipe](https://mediapipe.dev/)
- [PyAutoGUI](https://pyautogui.readthedocs.io/)
- [Pynput](https://pynput.readthedocs.io/)

---

Feel free to reach out if you have any questions or need further assistance. Happy coding!