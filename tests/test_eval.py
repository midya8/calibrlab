"""Deterministic fixture regression on the checked-in pipeline. Gates pin the
primary artifact (overconfidence before, calibrated after, exact accuracy
invariance); they are regression checks on this fixture, not general proofs."""
import unittest

from calibrlab.eval import run_evaluation


class TestEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = run_evaluation()

    def test_model_overfits_by_design(self):
        self.assertAlmostEqual(self.r["train_acc"], 1.0, places=9)
        self.assertLess(self.r["test_acc_raw"], 0.75)

    def test_raw_model_is_overconfident(self):
        self.assertGreaterEqual(self.r["ece_raw"], 0.15)

    def test_temperature_scaling_reduces_ece(self):
        self.assertGreaterEqual(self.r["ece_reduction"], 0.10)

    def test_accuracy_is_exactly_invariant(self):
        self.assertEqual(self.r["acc_drift"], 0.0)

    def test_nll_improves(self):
        self.assertLess(self.r["nll_cal"], self.r["nll_raw"])

    def test_overconfidence_implies_T_above_one(self):
        self.assertGreater(self.r["t_val"], 1.0)

    def test_validation_fit_lands_near_oracle(self):
        gap = abs(self.r["nll_cal"] - self.r["test_nll_oracle"])
        self.assertLessEqual(gap, 0.05)

    def test_determinism(self):
        r2 = run_evaluation()
        for k in ("ece_raw", "ece_cal", "nll_raw", "nll_cal", "t_val", "t_oracle"):
            self.assertEqual(self.r[k], r2[k])


if __name__ == "__main__":
    unittest.main()
