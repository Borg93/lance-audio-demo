"""Interactive AI-assist — the mock producer's shape contract.

The endpoint routes to a model server when MEDIA_ASSIST_URL is set, else this
deterministic mock so the draw/prompt→shapes round-trip is testable in-repo (proven
live via curl + playwright). These pin the shape shape the annotator renders.
"""

from __future__ import annotations

from backend.media_api.assist import AssistRequest, Region, _mock


def test_mock_boxes_the_drawn_region_labeled_with_the_prompt() -> None:
    shapes = _mock(
        AssistRequest(
            producer="grounding-dino",
            prompt="text line",
            region=Region(x=10.0, y=20.0, width=100.0, height=30.0),
        )
    )
    assert len(shapes) == 1
    s = shapes[0]
    assert s.shape_type == "rectangle"
    assert (s.x, s.y, s.width, s.height) == (10.0, 20.0, 100.0, 30.0)  # boxes the region
    assert s.label == "text line"  # labeled with the prompt
    assert s.confidence > 0


def test_mock_default_box_when_no_region() -> None:
    shapes = _mock(AssistRequest(prompt="figure"))
    assert len(shapes) == 1
    assert shapes[0].label == "figure"
    assert shapes[0].width > 0 and shapes[0].height > 0
