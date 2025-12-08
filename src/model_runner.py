import torch
import numpy as np
from share.gesture_model import AbstractGestureModel, GestureLabel
from share.utils import HandLandmark


class GestureModelRunner:
    def __init__(
        self, model_class: type[AbstractGestureModel], model_path: str, device: str
    ):
        self.device = device
        self.model = model_class()
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.landmarks_queue = []
        self.window_length = model_class.WINDOW_LENGTH

    def update_and_inference(self, landmarks: np.ndarray) -> GestureLabel:
        """
        Update the landmarks window and perform inference on the current window
        """
        assert landmarks.shape[0] == len(HandLandmark) and landmarks.shape[1] == 3

        # update
        self.landmarks_queue.append(landmarks)
        if len(self.landmarks_queue) > self.window_length:
            self.landmarks_queue.pop(0)

        # build landmarks window
        if len(self.landmarks_queue) < self.window_length:
            return GestureLabel.NONE
        landmarks_window = self.landmarks_queue[-self.window_length :]
        landmarks_window = np.stack(landmarks_window, axis=0)

        # inference
        with torch.no_grad():
            x_tensor = self.model.landmarks_window_to_X(landmarks_window)
            x_tensor = x_tensor.unsqueeze(0)  # add batch dimension
            x_tensor = x_tensor.to(next(self.model.parameters()).device)
            out = self.model.forward(x_tensor)
            pred_idx = out.argmax(dim=1).item()
            mappped_label = self.model.y_to_label(pred_idx)

        return mappped_label
