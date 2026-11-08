"""Five public synthetic controls; original training pipeline retained exactly."""
import json
from pathlib import Path

TEMPERATURES = [.1, .25, .5, .75, 1., 1.25, 1.5, 2., 2.5, 3., 3.5, 3.85, 4., 5., 6., 8., 10., 15., 20.]


def fixtures():
    return json.loads((Path(__file__).with_name("ledger_data") / "cases.json").read_text())


def build_bundle():
    """Regenerate all synthetic data with the untouched original trainer."""
    import math
    from .data import make_fixture
    from .mlp import MLP, train
    from .eval import run_evaluation
    f = make_fixture(seed=13)
    model = MLP(f["dim"], hidden=24, n_classes=f["n_classes"], seed=7)
    train(model, *f["train"], epochs=400, lr=.5)
    payload = {name: {"logits": model.predict_logits(f[split][0]), "labels": f[split][1]}
               for name, split in [("validation", "val"), ("test", "test")]}
    caller = {"validation": {"logits": [[4,0],[0,4],[4,0],[0,4]], "labels": [0,1,1,0]},
              "test": {"logits": [[3,0],[0,2]], "labels": [0,1]}}
    def same(z, y):
        return {name: {"logits": z, "labels": y} for name in ["validation", "test"]}
    def case(name, title, scope, p):
        return {"id": name, "title": title, "scope": scope, "payload": p}
    cases = [
        case("original", "Original six-class MLP", "Original seed13 training/logits and disjoint synthetic validation/test splits; descriptive fixture, not a benchmark.", payload),
        case("adverse", "Adverse caller example", "Original CLI example: selected validation T reaches10 and hurts test NLL; not a calibration success.", caller),
        case("ties", "Uniform ties", "Hand-constructed shared validation/test values. All logits equal, class0 is first maximum. Interface control, not independent evaluation.", same([[0,0,0]]*6, [0,1,2,0,1,2])),
        case("cancellation", "Within-bin cancellation", "Hand-constructed shared values. At T1, all confidences0.75; three correct and one wrong cancel signed gap. ECE can be zero without every row being accurate.", same([[math.log(3),0]]*4, [0,0,0,1])),
        case("extreme", "Singleton underflow", "Shared one-row numerical control. Wrong-class margin1000 has logit NLL1000 at T1 even when p(label) underflows to0; deleting the only row is undefined.", same([[1000,0]], [1]))]
    return {"cases": cases, "original_report": run_evaluation()}
