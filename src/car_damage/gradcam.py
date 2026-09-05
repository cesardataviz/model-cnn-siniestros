"""Grad-CAM (Selvaraju et al. 2020, Ec. 1-2), reutilizable para cualquier CNN
apuntando a su propia última capa convolucional. Es la herramienta de auditoría
para verificar que el modelo mira el daño real y no correlaciones espurias
(fondo, marca, placa) antes de considerar cualquier despliegue."""
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None):
        self.model.eval()
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        score = logits[:, class_idx]
        score.backward(retain_graph=True)

        alpha = self.gradients.mean(dim=(2, 3), keepdim=True)
        weighted = (alpha * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(weighted)

        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx, F.softmax(logits, dim=1).detach().cpu().numpy()[0]
