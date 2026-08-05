import unittest
from unittest.mock import patch

import torch

from src.data.violin_datamodule import ViolinCollator


class ViolinCollatorTest(unittest.TestCase):
    @staticmethod
    def _minibatch(batch_size: int) -> list[dict]:
        return [
            {
                "waveform": torch.ones(1, 4),
                "midi_tokens": torch.ones(1, dtype=torch.long),
                "tech_tokens": torch.ones(1, dtype=torch.long),
                "velocity_tokens": torch.ones(1, dtype=torch.long),
                "pos_midi": torch.ones(1, dtype=torch.long),
                "cc_tokens": torch.ones(1, 1),
                "pitch_shift": torch.tensor(0.0),
                "instrument_id": torch.tensor(0),
                "is_real": torch.tensor(True),
                "audio_path": f"audio-{index}",
                "midi_path": f"midi-{index}",
            }
            for index in range(batch_size)
        ]

    @staticmethod
    def _silent_count(batch: dict) -> int:
        return int((batch["waveform"] == 0).all(dim=(1, 2)).sum().item())

    @patch("src.data.violin_datamodule.random.sample", side_effect=lambda population, count: list(population)[:count])
    def test_fractional_expected_count_uses_stochastic_rounding(self, _sample) -> None:
        collator = ViolinCollator(silence_pair_prob=0.03)

        with patch("src.data.violin_datamodule.random.random", return_value=0.47):
            rounded_up = collator.collate(self._minibatch(16))
        with patch("src.data.violin_datamodule.random.random", return_value=0.48):
            rounded_down = collator.collate(self._minibatch(16))

        self.assertEqual(self._silent_count(rounded_up), 1)
        self.assertEqual(self._silent_count(rounded_down), 0)

    def test_integer_expected_count_is_exact(self) -> None:
        collator = ViolinCollator(silence_pair_prob=0.25)
        batch = collator.collate(self._minibatch(16))

        self.assertEqual(self._silent_count(batch), 4)

    def test_probability_must_be_between_zero_and_one(self) -> None:
        for probability in (-0.01, 1.01):
            with self.subTest(probability=probability):
                with self.assertRaises(ValueError):
                    ViolinCollator(silence_pair_prob=probability)


if __name__ == "__main__":
    unittest.main()