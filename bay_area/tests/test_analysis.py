import pytest
from bay_area.analyze import independent_points, interval


def test_icm_oracle_nontrivial_three_way_and_negative_control():
    # P1=1/2. P2=(3/10)*(5/7)+(2/10)*(5/8)=19/56.
    actual = independent_points([5,3,2], [], [], [50,30,20])
    assert actual[0] == pytest.approx(20 + 30/2 + 10*19/56)
    assert sum(actual) == pytest.approx(100)
    # Chip-proportional payout is the intentionally wrong scoring control.
    assert actual != pytest.approx([50,30,20])


def test_interval_uses_sample_variance_and_refuses_false_certainty():
    result = interval([0,2])
    assert result["mean"] == 1
    assert result["standard_error"] == pytest.approx(1)
    assert interval([1])["interval"] == [None,None]
