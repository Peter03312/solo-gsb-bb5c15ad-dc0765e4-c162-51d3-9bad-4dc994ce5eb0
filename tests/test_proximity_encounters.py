"""多车近距离相遇审计测试。

POST /api/v1/zones/{zone_id}/proximity-encounters：只在两车均有轨迹覆盖且
同时位于区内时比较同一时刻的平面距离；距离**严格小于**米制阈值的正时长
半开区间按无序车次对返回（含起止、持续秒数与区间最小距离）；相接且连续
满足条件的区间合并；瞬时触及忽略；区域与全部车次点集取自同一已提交读视图。
"""
import json
import threading

from tests.conftest import DAY_WINDOWS, SQUARE, register_square


def _batch(points):
    return {"points": [{"point_id": pid, "ts": ts, "x": x, "y": y} for pid, ts, x, y in points]}


def _upload(client, trip, points):
    r = client.post(f"/api/v1/trips/{trip}/points", json=_batch(points))
    assert r.status_code == 200, r.text
    return r


def _ask(client, zone_id, **overrides):
    body = {
        "trip_ids": ["T1", "T2"],
        "since": "2026-09-30T10:00:00Z",
        "until": "2026-09-30T10:10:00Z",
        "distance_threshold_m": 10.0,
    }
    body.update(overrides)
    return client.post(f"/api/v1/zones/{zone_id}/proximity-encounters", json=body)


# 正交横穿（均 1m/s）：T1 水平 y=40，T2 垂直 x=40；t∈[0,100]，t=40 在 (40,40) 相遇
CROSS_H = [
    ("a", "2026-09-30T10:00:00Z", 0, 40),
    ("b", "2026-09-30T10:01:40Z", 100, 40),
]
CROSS_V = [
    ("a", "2026-09-30T10:00:00Z", 40, 0),
    ("b", "2026-09-30T10:01:40Z", 40, 100),
]


def _pair_map(body):
    return {tuple(p["trip_ids"]): p["intervals"] for p in body["pairs"]}


# ---------------------------------------------------------------------
# 基本语义
# ---------------------------------------------------------------------

def test_basic_encounter_interval_and_min_distance(client):
    register_square(client, zone_id="pe1")
    _upload(client, "T1", CROSS_H)
    _upload(client, "T2", CROSS_V)

    r = _ask(client, "pe1", until="2026-09-30T10:02:00Z", distance_threshold_m=10)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["zone_id"] == "pe1"
    assert body["distance_threshold_m"] == 10.0
    assert body["since"] == "2026-09-30T10:00:00Z"
    assert body["until"] == "2026-09-30T10:02:00Z"
    assert "revision" not in body
    trips = {t["trip_id"]: t for t in body["trips"]}
    assert trips["T1"]["track_version"] == 1 and trips["T1"]["point_count"] == 2
    assert trips["T2"]["track_version"] == 1 and trips["T2"]["point_count"] == 2

    assert len(body["pairs"]) == 1
    pair = body["pairs"][0]
    assert pair["trip_ids"] == ["T1", "T2"]
    assert len(pair["intervals"]) == 1
    seg = pair["intervals"][0]
    # sqrt2*(40-t) < 10  ->  40-5sqrt2 < t < 40+5sqrt2
    expected = 10 * 2 ** 0.5
    assert abs(seg["duration_sec"] - expected) < 1e-6
    assert seg["min_distance_m"] == 0.0  # 交叉点在区内
    assert seg["start"] < "2026-09-30T10:00:40Z" < seg["end"]


def test_no_encounter_returns_empty_pairs(client):
    """合法但始终不够近：pairs 为空列表。"""
    register_square(client, zone_id="pe2")
    _upload(client, "T1", CROSS_H)
    # T2 恒在远处（穿越另一条 y=95 的线，最近纵向距离 55m）
    _upload(client, "T2", [
        ("a", "2026-09-30T10:00:00Z", 0, 95),
        ("b", "2026-09-30T10:01:40Z", 100, 95),
    ])
    r = _ask(client, "pe2", until="2026-09-30T10:02:00Z", distance_threshold_m=10)
    assert r.status_code == 200
    assert r.json()["pairs"] == []


def test_strict_less_than_threshold(client):
    """距离恰好等于阈值（恒距并行）不算相遇；略大于阈值整段满足。"""
    register_square(client, zone_id="pe3")
    # 恒距 10m 并行纵向，1m/s，从区外 40m 出发 -> 占用 [40,140)
    for tid, x in (("T1", 40), ("T2", 50)):
        _upload(client, tid, [
            ("a", "2026-09-30T10:00:00Z", x, -40),
            ("b", "2026-09-30T10:03:00Z", x, 140),
        ])

    # 恰好 10m：严格小于不成立 -> 空
    r = _ask(client, "pe3", distance_threshold_m=10,
             since="2026-09-30T09:00:00Z", until="2026-09-30T11:00:00Z")
    assert r.json()["pairs"] == []

    # 略大：整段共同占用 100s，最小距离即 10m
    r = _ask(client, "pe3", distance_threshold_m=10.5,
             since="2026-09-30T09:00:00Z", until="2026-09-30T11:00:00Z")
    segs = r.json()["pairs"][0]["intervals"]
    assert len(segs) == 1
    assert segs[0]["start"] == "2026-09-30T10:00:40Z"
    assert segs[0]["end"] == "2026-09-30T10:02:20Z"
    assert segs[0]["duration_sec"] == 100.0
    assert segs[0]["min_distance_m"] == 10.0


def test_instantaneous_touch_ignored(client):
    """最小距离恰好等于阈值的瞬时相切：无正时长区间，忽略。"""
    register_square(client, zone_id="pe4")
    # 两车沿同一直线同向：T2 在 T1 前方恒距 10m（等价瞬时触及阈值的退化场景由严格不等式覆盖）；
    # 这里构造相向运动且阈值等于最近距离 0 的反向情形——用恒距验证“相切无时长”。
    for tid, y in (("T1", 40), ("T2", 50)):
        _upload(client, tid, [
            ("a", "2026-09-30T10:00:00Z", 0, y),
            ("b", "2026-09-30T10:01:40Z", 100, y),
        ])
    # 阈值恰为恒距 10：任何时刻都不满足严格小于
    r = _ask(client, "pe4", until="2026-09-30T10:02:00Z", distance_threshold_m=10)
    assert r.json()["pairs"] == []


def test_abutting_intervals_merged(client):
    """相遇区间跨越某车轨迹点断点时，相接且连续满足的两段合并为一段。"""
    register_square(client, zone_id="pe5")
    # T1 在 t=45 处插入一个额外点（速度不变，断点落在相遇窗内）
    _upload(client, "T1", [
        ("a", "2026-09-30T10:00:00Z", 0, 40),
        ("m", "2026-09-30T10:00:45Z", 45, 40),
        ("b", "2026-09-30T10:01:40Z", 100, 40),
    ])
    _upload(client, "T2", CROSS_V)
    r = _ask(client, "pe5", until="2026-09-30T10:02:00Z", distance_threshold_m=10)
    segs = r.json()["pairs"][0]["intervals"]
    assert len(segs) == 1  # 断点 t=45 两侧相接 -> 合并
    assert abs(segs[0]["duration_sec"] - 10 * 2 ** 0.5) < 1e-6
    assert segs[0]["min_distance_m"] == 0.0


def test_hole_splits_intervals(client):
    """带洞边界规则沿用：两车在洞内（含洞边界）时间不比较，区间被洞切成两段。"""
    hole = [[[40, 40], [60, 40], [60, 60], [40, 60]]]
    register_square(client, zone_id="pe6", holes=hole)
    # 并排恒距 5m 沿 y=50 横穿，x 0->100 over 100s；洞 x∈[40,60] 对应 t∈[40,60]
    for tid, y in (("T1", 47.5), ("T2", 52.5)):
        _upload(client, tid, [
            ("a", "2026-09-30T10:00:00Z", 0, y),
            ("b", "2026-09-30T10:01:40Z", 100, y),
        ])
    r = _ask(client, "pe6", until="2026-09-30T10:02:00Z", distance_threshold_m=10)
    segs = r.json()["pairs"][0]["intervals"]
    assert [(s["start"], s["end"]) for s in segs] == [
        ("2026-09-30T10:00:00Z", "2026-09-30T10:00:40Z"),
        ("2026-09-30T10:01:00Z", "2026-09-30T10:01:40Z"),
    ]
    assert all(s["min_distance_m"] == 5.0 for s in segs)


def test_zone_window_clips_comparison(client):
    """区域生效窗口外不比较；半开窗口端点接触为零时长。"""
    register_square(
        client, zone_id="pe7",
        windows=[{"start": "2026-09-30T10:00:20Z", "end": "2026-09-30T10:00:50Z"}],
    )
    # 恒距 5m 并行横穿 y=50
    for tid, y in (("T1", 47.5), ("T2", 52.5)):
        _upload(client, tid, [
            ("a", "2026-09-30T10:00:00Z", -50, y),
            ("b", "2026-09-30T10:01:20Z", 150, y),
        ])
    r = _ask(client, "pe7", since="2026-09-30T10:00:00Z", until="2026-09-30T10:02:00Z",
             distance_threshold_m=10)
    segs = r.json()["pairs"][0]["intervals"]
    assert [(s["start"], s["end"]) for s in segs] == [
        ("2026-09-30T10:00:20Z", "2026-09-30T10:00:50Z"),
    ]


def test_query_window_halfopen_clips(client):
    """查询时间范围按半开 [since, until) 裁剪相遇区间。"""
    register_square(client, zone_id="pe8")
    _upload(client, "T1", CROSS_H)
    _upload(client, "T2", CROSS_V)
    r = _ask(client, "pe8", since="2026-09-30T10:00:40Z", until="2026-09-30T10:00:45Z",
             distance_threshold_m=20)
    segs = r.json()["pairs"][0]["intervals"]
    assert len(segs) == 1
    assert segs[0]["start"] == "2026-09-30T10:00:40Z"
    assert segs[0]["end"] == "2026-09-30T10:00:45Z"
    assert segs[0]["duration_sec"] == 5.0

    # until 恰好等于相遇起点（40-5sqrt2 ≈ 32.928932）：半开，零时长 -> 空
    r2 = _ask(client, "pe8", since="2026-09-30T09:00:00Z",
              until="2026-09-30T10:00:32.928932Z", distance_threshold_m=10)
    assert r2.json()["pairs"] == []


def test_both_must_be_inside_zone(client):
    """一辆在区外（即使物理距离很近）不比较：仅同时位于区内才计入。"""
    register_square(client, zone_id="pe9")
    # T1 沿区内 y=5 横穿；T2 沿区外 y=-5 同向并行，物理恒距 10m
    _upload(client, "T1", [
        ("a", "2026-09-30T10:00:00Z", 0, 5),
        ("b", "2026-09-30T10:01:40Z", 100, 5),
    ])
    _upload(client, "T2", [
        ("a", "2026-09-30T10:00:00Z", 0, -5),
        ("b", "2026-09-30T10:01:40Z", 100, -5),
    ])
    r = _ask(client, "pe9", until="2026-09-30T10:02:00Z", distance_threshold_m=20)
    assert r.json()["pairs"] == []


def test_trajectory_coverage_required(client):
    """某车轨迹覆盖结束后不再比较，即使其最后一点在区内。"""
    register_square(client, zone_id="pe10")
    _upload(client, "T1", [
        ("a", "2026-09-30T10:00:00Z", 0, 47.5),
        ("b", "2026-09-30T10:01:40Z", 100, 47.5),
    ])
    # T2 只覆盖到 t=30（点 (30,52.5) 在区内）
    _upload(client, "T2", [
        ("a", "2026-09-30T10:00:00Z", 0, 52.5),
        ("b", "2026-09-30T10:00:30Z", 30, 52.5),
    ])
    r = _ask(client, "pe10", until="2026-09-30T10:02:00Z", distance_threshold_m=10)
    segs = r.json()["pairs"][0]["intervals"]
    assert len(segs) == 1
    assert segs[0]["start"] == "2026-09-30T10:00:00Z"
    assert segs[0]["end"] == "2026-09-30T10:00:30Z"
    assert segs[0]["min_distance_m"] == 5.0


def test_single_point_trip_never_encounters(client):
    """单车点不足 2 无覆盖区间，合法请求返回空。"""
    register_square(client, zone_id="pe11")
    _upload(client, "T1", CROSS_H)
    _upload(client, "T2", [("solo", "2026-09-30T10:00:40Z", 40, 40)])
    r = _ask(client, "pe11", until="2026-09-30T10:02:00Z", distance_threshold_m=10)
    assert r.status_code == 200
    assert r.json()["pairs"] == []
    trips = {t["trip_id"]: t for t in r.json()["trips"]}
    assert trips["T2"]["point_count"] == 1


def test_multiple_unordered_pairs(client):
    """三车两两按无序车次对返回；车对按请求车次顺序规范化。"""
    register_square(client, zone_id="pe12")
    # T1/T2 在 y=40/42（恒距 2m），T3 远在 y=95
    _upload(client, "T1", [
        ("a", "2026-09-30T10:00:00Z", 0, 40),
        ("b", "2026-09-30T10:01:40Z", 100, 40),
    ])
    _upload(client, "T2", [
        ("a", "2026-09-30T10:00:00Z", 0, 42),
        ("b", "2026-09-30T10:01:40Z", 100, 42),
    ])
    _upload(client, "T3", [
        ("a", "2026-09-30T10:00:00Z", 0, 95),
        ("b", "2026-09-30T10:01:40Z", 100, 95),
    ])
    r = _ask(client, "pe12", trip_ids=["T1", "T2", "T3"],
             until="2026-09-30T10:02:00Z", distance_threshold_m=10)
    assert r.status_code == 200
    pairs = _pair_map(r.json())
    # 只有 (T1,T2) 相遇；无相遇的车对不出现
    assert set(pairs.keys()) == {("T1", "T2")}
    assert pairs[("T1", "T2")][0]["min_distance_m"] == 2.0

    # 顺序无关：颠倒请求顺序，车对键规范化为 [T1,T2]
    r2 = _ask(client, "pe12", trip_ids=["T3", "T2", "T1"],
              until="2026-09-30T10:02:00Z", distance_threshold_m=10)
    pairs2 = _pair_map(r2.json())
    assert set(pairs2.keys()) == {("T1", "T2")}
    assert pairs2[("T1", "T2")] == pairs[("T1", "T2")]


def test_min_distance_within_interval(client):
    """区间最小距离取该时间窗内的最小值（非整段最小值）。"""
    register_square(client, zone_id="pe13")
    _upload(client, "T1", CROSS_H)
    _upload(client, "T2", CROSS_V)
    # 裁到 t∈[40,45]：最小距离 0（在 t=40 边界）
    r = _ask(client, "pe13", since="2026-09-30T10:00:40Z", until="2026-09-30T10:00:45Z",
             distance_threshold_m=20)
    seg = r.json()["pairs"][0]["intervals"][0]
    assert seg["min_distance_m"] == 0.0
    # 裁到 t∈[41,46]：最小距离 sqrt2
    r2 = _ask(client, "pe13", since="2026-09-30T10:00:41Z", until="2026-09-30T10:00:46Z",
              distance_threshold_m=20)
    seg2 = r2.json()["pairs"][0]["intervals"][0]
    assert abs(seg2["min_distance_m"] - 2 ** 0.5) < 1e-6


# ---------------------------------------------------------------------
# 修订号 / 逐车轨迹版本复现
# ---------------------------------------------------------------------

def test_pinned_track_versions_reproduce(client):
    register_square(client, zone_id="pe14")
    _upload(client, "T1", CROSS_H)
    _upload(client, "T2", CROSS_V)
    # 更正 T1：b 点拉远 -> 当前不再相遇
    r = client.post(
        "/api/v1/trips/T1/corrections",
        json={
            "base_version": 1, "reason": "b 点坐标录错",
            "items": [{"action": "update", "point_id": "b",
                       "ts": "2026-09-30T10:01:40Z", "x": 100, "y": 500}],
        },
    )
    assert r.status_code == 200

    r = _ask(client, "pe14", until="2026-09-30T10:02:00Z", distance_threshold_m=10)
    assert r.json()["pairs"] == []

    r = _ask(client, "pe14", until="2026-09-30T10:02:00Z", distance_threshold_m=10,
             track_versions={"T1": 1})
    body = r.json()
    assert len(body["pairs"]) == 1
    assert abs(body["pairs"][0]["intervals"][0]["duration_sec"] - 10 * 2 ** 0.5) < 1e-6
    trips = {t["trip_id"]: t for t in body["trips"]}
    assert trips["T1"]["track_version"] == 1  # 指定回填
    assert trips["T2"]["track_version"] == 1  # 未指定取当前


def test_pinned_revision_reproduces_zone(client):
    hole = [[[40, 40], [60, 40], [60, 60], [40, 60]]]
    register_square(client, zone_id="pe15", holes=hole)   # rev1 带洞
    register_square(client, zone_id="pe15")               # rev2 无洞
    for tid, y in (("T1", 47.5), ("T2", 52.5)):
        _upload(client, tid, [
            ("a", "2026-09-30T10:00:00Z", 0, y),
            ("b", "2026-09-30T10:01:40Z", 100, y),
        ])
    r = _ask(client, "pe15", until="2026-09-30T10:02:00Z", distance_threshold_m=10,
             revision=1)
    body = r.json()
    assert body["revision"] == 1
    segs = body["pairs"][0]["intervals"]
    assert [(s["start"], s["end"]) for s in segs] == [
        ("2026-09-30T10:00:00Z", "2026-09-30T10:00:40Z"),
        ("2026-09-30T10:01:00Z", "2026-09-30T10:01:40Z"),
    ]
    # 最新（无洞）为整段
    r2 = _ask(client, "pe15", until="2026-09-30T10:02:00Z", distance_threshold_m=10)
    assert "revision" not in r2.json()
    assert len(r2.json()["pairs"][0]["intervals"]) == 1


def test_revision_and_track_versions_combine(client):
    register_square(client, zone_id="pe16")
    register_square(client, zone_id="pe16",
                    exterior=[[0, 0], [200, 0], [200, 200], [0, 200]])  # rev2
    _upload(client, "T1", CROSS_H)
    _upload(client, "T2", CROSS_V)
    r = _ask(client, "pe16", until="2026-09-30T10:02:00Z", distance_threshold_m=10,
             revision=1, track_versions={"T1": 1, "T2": 1})
    body = r.json()
    assert body["revision"] == 1
    assert len(body["pairs"]) == 1


def test_latest_state_used_when_version_omitted(client):
    """未指定版本的车取查询时最新已提交状态：追加新点改变轨迹后立即生效。"""
    register_square(client, zone_id="pe17")
    _upload(client, "T1", [
        ("a", "2026-09-30T10:00:00Z", 0, 40),
        ("b", "2026-09-30T10:01:40Z", 100, 40),
    ])
    # T2 初始静止在远处（两点同坐标 y=95）
    _upload(client, "T2", [
        ("a", "2026-09-30T10:00:00Z", 40, 95),
        ("b", "2026-09-30T10:02:00Z", 40, 95),
    ])
    r = _ask(client, "pe17", until="2026-09-30T10:02:00Z", distance_threshold_m=10)
    assert r.json()["pairs"] == []

    # 更正 T2 为垂直穿线（版本 2）：开始与 T1 相遇
    client.post(
        "/api/v1/trips/T2/corrections",
        json={"base_version": 1, "reason": "坐标录错", "items": [
            {"action": "update", "point_id": "a", "ts": "2026-09-30T10:00:00Z", "x": 40, "y": 0},
            {"action": "update", "point_id": "b", "ts": "2026-09-30T10:01:40Z", "x": 40, "y": 100},
        ]},
    )
    r = _ask(client, "pe17", until="2026-09-30T10:02:00Z", distance_threshold_m=10)
    assert len(r.json()["pairs"]) == 1
    trips = {t["trip_id"]: t for t in r.json()["trips"]}
    assert trips["T2"]["track_version"] == 2


# ---------------------------------------------------------------------
# 错误：区域 / 车次 / 版本不存在
# ---------------------------------------------------------------------

def test_zone_not_found(client):
    _upload(client, "T1", CROSS_H)
    _upload(client, "T2", CROSS_V)
    r = _ask(client, "ghost-zone", until="2026-09-30T10:02:00Z")
    assert r.status_code == 404
    assert "危险区不存在" in r.json()["detail"]


def test_zone_not_registered_at_revision(client):
    register_square(client, zone_id="pe-z")
    _upload(client, "T1", CROSS_H)
    _upload(client, "T2", CROSS_V)
    r = _ask(client, "pe-z", revision=999, until="2026-09-30T10:02:00Z")
    assert r.status_code == 404
    assert "超出当前范围" in r.json()["detail"]
    register_square(client, zone_id="other")  # rev2
    r = _ask(client, "other", revision=1, until="2026-09-30T10:02:00Z")
    assert r.status_code == 404
    assert "尚未登记" in r.json()["detail"]


def test_trip_not_found(client):
    register_square(client, zone_id="pe-t")
    _upload(client, "T1", CROSS_H)
    r = _ask(client, "pe-t", trip_ids=["T1", "ghost"], until="2026-09-30T10:02:00Z")
    assert r.status_code == 404
    assert "车次不存在" in r.json()["detail"]
    r = _ask(client, "pe-t", trip_ids=["T1", "ghost"], until="2026-09-30T10:02:00Z",
             track_versions={"ghost": 1})
    assert r.status_code == 404
    assert "车次不存在" in r.json()["detail"]


def test_track_version_out_of_range(client):
    register_square(client, zone_id="pe-v")
    _upload(client, "T1", CROSS_H)
    _upload(client, "T2", CROSS_V)
    r = _ask(client, "pe-v", track_versions={"T1": 0}, until="2026-09-30T10:02:00Z")
    assert r.status_code == 404 and "基线" in r.json()["detail"]
    r = _ask(client, "pe-v", track_versions={"T2": 9}, until="2026-09-30T10:02:00Z")
    assert r.status_code == 404 and "晚于" in r.json()["detail"]


# ---------------------------------------------------------------------
# 错误：非法阈值 / 时间 / 重复车次 / 请求形状
# ---------------------------------------------------------------------

def test_invalid_threshold_rejected(client):
    register_square(client, zone_id="pe-th")
    _upload(client, "T1", CROSS_H)
    _upload(client, "T2", CROSS_V)
    for bad in (0, -1, -0.0001):
        r = _ask(client, "pe-th", distance_threshold_m=bad, until="2026-09-30T10:02:00Z")
        assert r.status_code == 400, (bad, r.status_code)
        assert "distance_threshold_m" in r.json()["detail"]
    # 非数值
    r = _ask(client, "pe-th", distance_threshold_m="ten", until="2026-09-30T10:02:00Z")
    assert r.status_code in (400, 422)
    # 布尔
    r = _ask(client, "pe-th", distance_threshold_m=True, until="2026-09-30T10:02:00Z")
    assert r.status_code in (400, 422)


def test_nonfinite_threshold_rejected_via_raw_json(client):
    """NaN/Infinity 通过原始 JSON token 传入时稳定 422（不产生 500）。"""
    register_square(client, zone_id="pe-nf")
    base = {
        "trip_ids": ["T1", "T2"],
        "since": "2026-09-30T10:00:00Z",
        "until": "2026-09-30T10:02:00Z",
        "distance_threshold_m": 10,
    }
    for token in ("NaN", "Infinity", "-Infinity"):
        raw = json.dumps(base)[:-1] + f', "distance_threshold_m": {token}}}'
        r = client.post(
            "/api/v1/zones/pe-nf/proximity-encounters",
            content=raw, headers={"Content-Type": "application/json"},
        )
        assert r.status_code in (400, 422), (token, r.status_code, r.text)


def test_duplicate_trip_ids_rejected(client):
    register_square(client, zone_id="pe-d")
    r = _ask(client, "pe-d", trip_ids=["T1", "T1"])
    assert r.status_code == 400 and "重复" in r.json()["detail"]


def test_invalid_time_range_rejected(client):
    register_square(client, zone_id="pe-r")
    r = _ask(client, "pe-r", since="2026-09-30T10:02:00Z", until="2026-09-30T10:02:00Z")
    assert r.status_code == 400
    r = _ask(client, "pe-r", since="2026-09-30T10:00:00", until="2026-09-30T10:02:00Z")
    assert r.status_code == 400 and "时区" in r.json()["detail"]
    r = _ask(client, "pe-r", since="nope")
    assert r.status_code == 400


def test_invalid_request_shape_rejected(client):
    register_square(client, zone_id="pe-s")
    r = _ask(client, "pe-s", trip_ids=["T1"])
    assert r.status_code in (400, 422)
    r = client.post("/api/v1/zones/pe-s/proximity-encounters",
                    json={"trip_ids": ["T1", "T2"],
                          "since": "2026-09-30T10:00:00Z"})  # 缺 until/threshold
    assert r.status_code == 422
    r = _ask(client, "pe-s", bogus=1)
    assert r.status_code == 422


def test_invalid_track_versions_rejected(client):
    register_square(client, zone_id="pe-tv")
    _upload(client, "T1", CROSS_H)
    _upload(client, "T2", CROSS_V)
    r = _ask(client, "pe-tv", track_versions={"T1": -1}, until="2026-09-30T10:02:00Z")
    assert r.status_code in (400, 422)
    r = _ask(client, "pe-tv", track_versions={"T1": "x"}, until="2026-09-30T10:02:00Z")
    assert r.status_code in (400, 422)
    r = _ask(client, "pe-tv", track_versions={"T9": 1}, until="2026-09-30T10:02:00Z")
    assert r.status_code == 400 and "T9" in r.json()["detail"]


# ---------------------------------------------------------------------
# 同一已提交读视图：并发更新不拼出不存在的组合
# ---------------------------------------------------------------------

def test_concurrent_writes_use_single_read_view(client):
    register_square(client, zone_id="pe-live")
    _upload(client, "T1", CROSS_H)
    _upload(client, "T2", CROSS_V)

    stop = threading.Event()
    failures = []

    def writer() -> None:
        i = 0
        while not stop.is_set():
            client.post(
                "/api/v1/zones",
                json={"zone_id": "pe-live",
                      "exterior": SQUARE, "holes": [], "windows": DAY_WINDOWS},
            )
            client.post(
                "/api/v1/trips/T1/points",
                json={"points": [
                    {"point_id": f"extra-{i}", "ts": f"2026-09-30T11:{i % 60:02d}:00Z",
                     "x": 200 + i, "y": 200},
                ]},
            )
            i += 1

    def reader() -> None:
        pinned = dict(
            until="2026-09-30T10:02:00Z",
            distance_threshold_m=10,
            revision=1, track_versions={"T1": 1, "T2": 1},
        )
        while not stop.is_set():
            r = _ask(client, "pe-live", **pinned)
            if r.status_code != 200:
                failures.append(("status", r.status_code, r.text))
                return
            body = r.json()
            trips = {t["trip_id"]: t for t in body["trips"]}
            if trips["T1"]["track_version"] != 1 or trips["T2"]["track_version"] != 1:
                failures.append(("version-mix", body))
                return
            # 固定版本组合：结果确定可复现
            r2 = _ask(client, "pe-live", **pinned)
            if r2.json() != body:
                failures.append(("nondeterministic",))
                return
            segs = body["pairs"][0]["intervals"]
            if abs(segs[0]["duration_sec"] - 10 * 2 ** 0.5) > 1e-6:
                failures.append(("wrong-duration", segs))
                return

    threads = [threading.Thread(target=writer)] + [threading.Thread(target=reader) for _ in range(2)]
    for t in threads:
        t.start()
    stop.wait(timeout=1.0)
    stop.set()
    for t in threads:
        t.join(timeout=10)
    assert not failures


def test_existing_apis_unaffected(client):
    """新接口上线后，既有穿越/占用查询语义不变。"""
    register_square(client, zone_id="pe-keep")
    _upload(client, "T1", CROSS_H)
    r = client.get("/api/v1/trips/T1/crossings?zone_id=pe-keep")
    assert r.status_code == 200
    assert r.json()["zones"][0]["total_duration_sec"] == 100.0
    r2 = client.post("/api/v1/zones/pe-keep/co-occupancy", json={
        "trip_ids": ["T1"],
        "since": "2026-09-30T10:00:00Z", "until": "2026-09-30T10:02:00Z",
        "threshold": 2,
    })
    assert r2.status_code in (400, 422)
