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


def test_sam_segments_the_drawn_box_as_a_polygon() -> None:
    shapes = _mock(
        AssistRequest(producer="sam-click", region=Region(x=40.0, y=40.0, width=160.0, height=90.0))
    )
    assert len(shapes) == 1
    s = shapes[0]
    assert s.shape_type == "polygon"  # a mask, not a box
    assert len(s.polygon) == 8  # a flat [x,y,...] quad within the region
    assert s.label == "object"  # SAM needs no prompt
    assert s.confidence > 0


def test_sam_click_grows_a_patch_around_the_point() -> None:
    # A click commits as a zero-size region — the patch is centered on the point.
    shapes = _mock(
        AssistRequest(producer="sam-click", region=Region(x=100.0, y=100.0, width=0.0, height=0.0))
    )
    s = shapes[0]
    assert s.width > 1 and s.height > 1  # grown, not zero
    assert s.x < 100.0 and s.y < 100.0  # centered on the clicked point
