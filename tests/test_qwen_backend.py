"""Constructor regression checks without downloading model or spaCy weights."""
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class QwenBackendTests(unittest.TestCase):
    def test_selected_backend_reaches_pretrained_loader(self):
        with patch("spacy.load"):
            from ZoomEye.zoom_model_qwenvl import ZoomModelQwenVL
        for backend in (None, "sdpa", "eager", "flash_attention_2"):
            with self.subTest(backend=backend):
                processor = MagicMock()
                processor.image_processor.image_mean = [.5, .5, .5]
                processor.tokenizer.return_value = SimpleNamespace(input_ids=[1])
                with patch("ZoomEye.zoom_model_qwenvl.AutoProcessor.from_pretrained", return_value=processor), \
                     patch("ZoomEye.zoom_model_qwenvl.Qwen2_5_VLForConditionalGeneration.from_pretrained") as load:
                    kwargs = {} if backend is None else {"attn_implementation": backend}
                    ZoomModelQwenVL("unused-local-path", device="cpu", **kwargs)
                self.assertEqual(load.call_args.kwargs["attn_implementation"], backend or "flash_attention_2")
                self.assertEqual(load.call_args.kwargs["device_map"], "cpu")


if __name__ == "__main__":
    unittest.main()
