"""危险区内近距离相遇复核测试。

POST /api/v1/zones/{zone_id}/proximity：复核指定危险区内两车是否曾近距离相遇。
区域与全部车次点集从同一已提交读视图读取；只在两车均有轨迹覆盖且同时位于区内时
比较同一时刻的平面距离；按无序车次对返回距离严格小于阈值的正时长区间
（起止、持续秒数、区间最小距离），相接且连续满足条件的区间合并，瞬时触及忽略；
合法但无相遇返回空列表。
"""
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
        "until": "2026-09-30T10:05:00Z",
        "max_distance": 25,
    }
    body.update(overrides)
    return client.post(f"/api/v1/zones/{zone_id}/proximity", json=body)


# T1 沿 y=40 以 1m/s 横穿方块：A(t) = (t, 40)，t ∈ [0, 100]（10:00:00 起）
TRACK_X40 = [
    ("a", "2026-09-30T10:00:00Z", 0, 40),
    ("b", "2026-09-30T10:01:40Z", 100, 40),
]
# T2 沿 x=t 从 (0,90) 到 (100,50)：B(t) = (t, 90-0.4t)，与 T1 距离 |0.4t-50|
TRACK_CONV = [
    ("a", "2026-09-30T10:00:00Z", 0, 90),
    ("b", "2026-09-30T10:01:40Z", 100, 50),
]
# T2 与 T1 平行、恒定相距 20m
TRACK_PAR20 = [
    ("a", "2026-09-30T10:00:00Z", 0, 60),
    ("b", "2026-09-30T10:01:40Z", 100, 60),
]


# ---------------------------------------------------------------------
# 基本语义
# ---------------------------------------------------------------------

def test_basic_converging_pair(client):
    """距离 |0.4t-50| < 25 → t ∈ (62.5, 187.5)，覆盖 [0,100] → [62.5,100)，最小距离 10。"""
    register_square(client, zone_id="px1")
    _upload(client, "T1", TRACK_X40)
    _upload(client, "T2", TRACK_CONV)

    r = _ask(client, "px1")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["zone_id"] == "px1"
    assert body["since"] == "2026-09-30T10:00:00Z"
    assert body["until"] == "2026-09-30T10:05:00Z"
    assert body["max_distance"] == 25.0
    assert "revision" not in body  # 未指定修订号：不出现
    trips = {t["trip_id"]: t for t in body["trips"]}
    assert trips["T1"]["track_version"] == 1 and trips["T1"]["point_count"] == 2
    assert trips["T2"]["track_version"] == 1 and trips["T2"]["point_count"] == 2

    assert len(body["pairs"]) == 1
    pair = body["pairs"][0]
    assert pair["trip_ids"] == ["T1", "T2"]
    assert len(pair["intervals"]) == 1
    seg = pair["intervals"][0]
    assert seg["start"] == "2026-09-30T10:01:02.500000Z"
    assert seg["end"] == "2026-09-30T10:01:40Z"
    assert seg["duration_sec"] == 37.5
    assert seg["min_distance"] == 10.0


def test_min_distance_at_interior_vertex(client):
    """相对运动最近点在区间内部：最小距离为抛物线顶点处取值（此处相撞为 0）。"""
    register_square(client, zone_id="px2")
    _upload(client, "T1", TRACK_X40)
    # B(t) = (t, 70-0.4t)：r_y = 0.4t-30，|r|<25 → t ∈ (12.5, 137.5)；t=75 时距离 0
    _upload(client, "T2", [
        ("a", "2026-09-30T10:00:00Z", 0, 70),
        ("b", "2026-09-30T10:01:40Z", 100, 30),
    ])
    r = _ask(client, "px2")
    assert r.status_code == 200, r.text
    seg = r.json()["pairs"][0]["intervals"][0]
    assert seg["start"] == "2026-09-30T10:00:12.500000Z"
    assert seg["end"] == "2026-09-30T10:01:40Z"
    assert seg["duration_sec"] == 87.5
    assert seg["min_distance"] == 0.0


def test_constant_distance_and_strict_threshold(client):
    """恒定距离 20m：阈值 25 计入整段；阈值恰为 20（严格小于不满足）不计。"""
    register_square(client, zone_id="px3")
    _upload(client, "T1", TRACK_X40)
    _upload(client, "T2", TRACK_PAR20)

    r = _ask(client, "px3", max_distance=25)
    seg = r.json()["pairs"][0]["intervals"][0]
    assert seg["start"] == "2026-09-30T10:00:00Z"
    assert seg["end"] == "2026-09-30T10:01:40Z"
    assert seg["duration_sec"] == 100.0
    assert seg["min_distance"] == 20.0

    # 距离 == 阈值：严格小于不成立 → 空
    r = _ask(client, "px3", max_distance=20)
    assert r.status_code == 200
    assert r.json()["pairs"] == []
    # 阈值小于距离：空
    r = _ask(client, "px3", max_distance=15)
    assert r.json()["pairs"] == []
    # 阈值略大于距离：整段计入
    r = _ask(client, "px3", max_distance=20.000001)
    assert len(r.json()["pairs"][0]["intervals"]) == 1


def test_instantaneous_touch_ignored(client):
    """距离仅在某一瞬等于阈值（相切）：零时长触及忽略；阈值稍大则得正时长区间。"""
    register_square(client, zone_id="px4")
    _upload(client, "T1", TRACK_X40)
    # B：y 从 90 降到 65（t=50）再升回 90；与 T1 最小距离恰为 25（t=50 一瞬）
    _upload(client, "T2", [
        ("a", "2026-09-30T10:00:00Z", 0, 90),
        ("b", "2026-09-30T10:00:50Z", 50, 65),
        ("c", "2026-09-30T10:01:40Z", 100, 90),
    ])
    r = _ask(client, "px4", max_distance=25)
    assert r.status_code == 200
    assert r.json()["pairs"] == []

    # 阈值 30：|r|<30 → t ∈ (40, 60)，跨越中间点时刻相接合并为一段，最小距离 25
    r = _ask(client, "px4", max_distance=30)
    segs = r.json()["pairs"][0]["intervals"]
    assert len(segs) == 1
    assert segs[0]["start"] == "2026-09-30T10:00:40Z"
    assert segs[0]["end"] == "2026-09-30T10:01:00Z"
    assert segs[0]["duration_sec"] == 20.0
    assert segs[0]["min_distance"] == 25.0


def test_merge_across_point_timestamp(client):
    """条件在一车的轨迹点时刻两侧连续满足：相接区间合并为一段。"""
    register_square(client, zone_id="px5")
    # T1 中间多一个共线点（t=50 处插值连续）
    _upload(client, "T1", [
        ("a", "2026-09-30T10:00:00Z", 0, 40),
        ("m", "2026-09-30T10:00:50Z", 50, 40),
        ("b", "2026-09-30T10:01:40Z", 100, 40),
    ])
    _upload(client, "T2", TRACK_PAR20)
    r = _ask(client, "px5")
    segs = r.json()["pairs"][0]["intervals"]
    assert len(segs) == 1
    assert segs[0]["start"] == "2026-09-30T10:00:00Z"
    assert segs[0]["end"] == "2026-09-30T10:01:40Z"
    assert segs[0]["duration_sec"] == 100.0
    assert segs[0]["min_distance"] == 20.0


def test_separate_intervals_not_merged(client):
    """距离先低于阈值、再高于、再低于：得到两个不相接区间，各自给出最小距离。"""
    register_square(client, zone_id="px6")
    _upload(client, "T1", TRACK_X40)
    # B_y：45 → 90 → 45；|r_y| = 5+0.9t（前半）/ 后半对称，<25 → t<200/9 与 t>70/0.9+50
    _upload(client, "T2", [
        ("a", "2026-09-30T10:00:00Z", 0, 45),
        ("b", "2026-09-30T10:00:50Z", 50, 90),
        ("c", "2026-09-30T10:01:40Z", 100, 45),
    ])
    r = _ask(client, "px6")
    segs = r.json()["pairs"][0]["intervals"]
    assert [(s["start"], s["end"], s["duration_sec"], s["min_distance"]) for s in segs] == [
        ("2026-09-30T10:00:00Z", "2026-09-30T10:00:22.222222Z", 22.222222, 5.0),
        ("2026-09-30T10:01:17.777778Z", "2026-09-30T10:01:40Z", 22.222222, 5.0),
    ]


def test_no_encounter_returns_empty_list(client):
    """合法查询但任何时刻距离都不小于阈值：pairs 为空列表（HTTP 200）。"""
    register_square(client, zone_id="px7")
    _upload(client, "T1", TRACK_X40)
    _upload(client, "T2", [
        ("a", "2026-09-30T10:00:00Z", 0, 90),
        ("b", "2026-09-30T10:01:40Z", 100, 90),
    ])  # 恒定相距 50m
    r = _ask(client, "px7")
    assert r.status_code == 200
    assert r.json()["pairs"] == []
    # trips 仍回填实际使用的版本与点数
    assert len(r.json()["trips"]) == 2


def test_close_outside_zone_not_counted(client):
    """只在两车同时位于区内时才比较：一车贴近区外行驶（相距 1m）不算相遇。"""
    register_square(client, zone_id="px8")
    _upload(client, "T1", [
        ("a", "2026-09-30T10:00:00Z", 0, 99.5),
        ("b", "2026-09-30T10:01:40Z", 100, 99.5),
    ])
    _upload(client, "T2", [
        ("a", "2026-09-30T10:00:00Z", 0, 100.5),  # 区外 1m
        ("b", "2026-09-30T10:01:40Z", 100, 100.5),
    ])
    r = _ask(client, "px8", max_distance=5)
    assert r.status_code == 200
    assert r.json()["pairs"] == []


def test_hole_excludes_encounter(client):
    """带洞边界规则沿用：一车穿洞期间不比较，相遇区间被洞切成两段。"""
    hole = [[[40, 40], [60, 40], [60, 60], [40, 60]]]
    register_square(client, zone_id="px9", holes=hole)
    _upload(client, "T1", [
        ("a", "2026-09-30T10:00:00Z", 0, 50),   # 横穿洞（y=50 ∈ 洞 y 范围）
        ("b", "2026-09-30T10:01:40Z", 100, 50),
    ])
    _upload(client, "T2", [
        ("a", "2026-09-30T10:00:00Z", 0, 70),   # y=70 在洞上方，覆盖全程
        ("b", "2026-09-30T10:01:40Z", 100, 70),
    ])
    # T1 覆盖 [0,40] 与 [60,100]（洞及其边界不计），T2 覆盖全程；恒定相距 20m
    r = _ask(client, "px9", max_distance=25)
    segs = r.json()["pairs"][0]["intervals"]
    assert [(s["start"], s["end"], s["duration_sec"]) for s in segs] == [
        ("2026-09-30T10:00:00Z", "2026-09-30T10:00:40Z", 40.0),
        ("2026-09-30T10:01:00Z", "2026-09-30T10:01:40Z", 40.0),
    ]
    assert all(s["min_distance"] == 20.0 for s in segs)


def test_zone_window_clips_encounter(client):
    """区域 UTC 半开生效窗口沿用：只统计窗口内的相遇。"""
    register_square(
        client, zone_id="px10",
        windows=[{"start": "2026-09-30T10:00:30Z", "end": "2026-09-30T10:01:10Z"}],
    )
    _upload(client, "T1", TRACK_X40)
    _upload(client, "T2", TRACK_CONV)
    r = _ask(client, "px10")
    segs = r.json()["pairs"][0]["intervals"]
    # 相遇 (62.5,100) ∩ 窗口 [30,70) → [62.5,70)
    assert len(segs) == 1
    assert segs[0]["start"] == "2026-09-30T10:01:02.500000Z"
    assert segs[0]["end"] == "2026-09-30T10:01:10Z"
    assert segs[0]["duration_sec"] == 7.5
    assert segs[0]["min_distance"] == 22.0  # t=70 时 |0.4*70-50| = 22


def test_query_range_clips_encounter(client):
    """查询范围 [since, until) 半开裁剪相遇区间。"""
    register_square(client, zone_id="px11")
    _upload(client, "T1", TRACK_X40)
    _upload(client, "T2", TRACK_CONV)
    r = _ask(client, "px11", since="2026-09-30T10:01:20Z", until="2026-09-30T10:05:00Z")
    seg = r.json()["pairs"][0]["intervals"][0]
    assert seg["start"] == "2026-09-30T10:01:20Z"
    assert seg["end"] == "2026-09-30T10:01:40Z"
    assert seg["duration_sec"] == 20.0
    assert seg["min_distance"] == 10.0

    # until 早于相遇开始：空
    r = _ask(client, "px11", since="2026-09-30T10:00:00Z", until="2026-09-30T10:01:00Z")
    assert r.json()["pairs"] == []


def test_single_point_trip_never_meets(client):
    """点数不足 2 的车次无轨迹覆盖，不产生相遇，但请求本身合法。"""
    register_square(client, zone_id="px12")
    _upload(client, "T1", TRACK_X40)
    _upload(client, "T2", [("solo", "2026-09-30T10:00:30Z", 50, 45)])
    r = _ask(client, "px12", max_distance=100)
    assert r.status_code == 200
    assert r.json()["pairs"] == []
    trips = {t["trip_id"]: t for t in r.json()["trips"]}
    assert trips["T2"]["point_count"] == 1


def test_multiple_trip_pairs(client):
    """多车次按无序对返回：只有距离达标的对出现；同一对只返回一次。"""
    register_square(client, zone_id="px13")
    _upload(client, "T1", TRACK_X40)
    _upload(client, "T2", TRACK_PAR20)                    # 距 T1 20m
    _upload(client, "T3", [
        ("a", "2026-09-30T10:00:00Z", 0, 95),             # 距 T2 35m、距 T1 55m
        ("b", "2026-09-30T10:01:40Z", 100, 95),
    ])
    r = _ask(client, "px13", trip_ids=["T1", "T2", "T3"], max_distance=25)
    body = r.json()
    assert len(body["pairs"]) == 1
    assert body["pairs"][0]["trip_ids"] == ["T1", "T2"]

    # 放宽阈值后 (T2,T3)（35m）也进入；同一无序对只出现一次，按请求车次顺序枚举
    r = _ask(client, "px13", trip_ids=["T3", "T2", "T1"], max_distance=40)
    pairs = r.json()["pairs"]
    assert [p["trip_ids"] for p in pairs] == [["T3", "T2"], ["T2", "T1"]]

    # 阈值 60 覆盖全部三对（T1-T3 相距 55m）
    r = _ask(client, "px13", trip_ids=["T3", "T2", "T1"], max_distance=60)
    pairs = r.json()["pairs"]
    assert [p["trip_ids"] for p in pairs] == [["T3", "T2"], ["T3", "T1"], ["T2", "T1"]]


# ---------------------------------------------------------------------
# 逐车轨迹版本与区域修订号
# ---------------------------------------------------------------------

def test_pinned_track_versions_reproduce(client):
    """逐车轨迹版本：按指定版本点集核算；未指定的车取查询时最新已提交状态。"""
    register_square(client, zone_id="px14")
    _upload(client, "T1", TRACK_X40)
    _upload(client, "T2", TRACK_CONV)
    # 更正 T1：b 点改写到 (100,10) → 与 T2 距离恒 > 25，当前状态无相遇
    r = client.post(
        "/api/v1/trips/T1/corrections",
        json={
            "base_version": 1,
            "reason": "b 点坐标录错",
            "items": [{"action": "update", "point_id": "b",
                       "ts": "2026-09-30T10:01:40Z", "x": 100, "y": 10}],
        },
    )
    assert r.status_code == 200

    r = _ask(client, "px14")
    assert r.json()["pairs"] == []

    # 指定 T1@1：按当时保留的点集复现 → 相遇 [62.5,100)
    r = _ask(client, "px14", track_versions={"T1": 1})
    body = r.json()
    seg = body["pairs"][0]["intervals"][0]
    assert seg["duration_sec"] == 37.5
    assert seg["min_distance"] == 10.0
    trips = {t["trip_id"]: t for t in body["trips"]}
    assert trips["T1"]["track_version"] == 1   # 指定的版本回填
    assert trips["T2"]["track_version"] == 1   # 未指定：查询时当前版本回填


def test_pinned_revision_reproduces_zone_config(client):
    """区域修订号：按截至该号的区域配置核算，响应回填 revision。"""
    register_square(client, zone_id="px15")  # rev1: [0,100]x[0,100]
    register_square(client, zone_id="px15",
                    exterior=[[0, 0], [30, 0], [30, 30], [0, 30]])  # rev2: 两车都在区外
    _upload(client, "T1", TRACK_X40)
    _upload(client, "T2", TRACK_CONV)

    r = _ask(client, "px15")
    body = r.json()
    assert "revision" not in body
    assert body["pairs"] == []

    r = _ask(client, "px15", revision=1)
    body = r.json()
    assert body["revision"] == 1
    seg = body["pairs"][0]["intervals"][0]
    assert seg["duration_sec"] == 37.5
    assert seg["min_distance"] == 10.0


def test_revision_and_track_versions_combine(client):
    """区域修订号与逐车轨迹版本可同时指定。"""
    register_square(client, zone_id="px16")  # rev1
    register_square(client, zone_id="px16",
                    exterior=[[0, 0], [30, 0], [30, 30], [0, 30]])  # rev2
    _upload(client, "T1", TRACK_X40)
    _upload(client, "T2", TRACK_CONV)
    r = client.post(
        "/api/v1/trips/T1/corrections",
        json={
            "base_version": 1,
            "reason": "b 点坐标录错",
            "items": [{"action": "update", "point_id": "b",
                       "ts": "2026-09-30T10:01:40Z", "x": 100, "y": 10}],
        },
    )
    assert r.status_code == 200

    r = _ask(client, "px16", revision=1, track_versions={"T1": 1})
    body = r.json()
    assert body["revision"] == 1
    assert body["pairs"][0]["intervals"][0]["duration_sec"] == 37.5
    trips = {t["trip_id"]: t for t in body["trips"]}
    assert trips["T1"]["track_version"] == 1
    assert trips["T2"]["track_version"] == 1


def test_latest_committed_state_used_when_version_omitted(client):
    """未指定版本的车取查询时最新已提交状态：新点上传后立即生效。"""
    register_square(client, zone_id="px17")
    _upload(client, "T1", TRACK_X40)
    _upload(client, "T2", [("a", "2026-09-30T10:00:00Z", 0, 90)])
    r = _ask(client, "px17")
    assert r.json()["pairs"] == []  # T2 仅一点，无覆盖

    _upload(client, "T2", [("b", "2026-09-30T10:01:40Z", 100, 50)])
    r = _ask(client, "px17")
    body = r.json()
    assert len(body["pairs"]) == 1
    trips = {t["trip_id"]: t for t in body["trips"]}
    assert trips["T2"]["track_version"] == 2


# ---------------------------------------------------------------------
# 错误：区域 / 车次 / 版本不存在
# ---------------------------------------------------------------------

def test_zone_not_found(client):
    _upload(client, "T1", TRACK_X40)
    _upload(client, "T2", TRACK_CONV)
    r = _ask(client, "ghost-zone")
    assert r.status_code == 404
    assert "危险区不存在" in r.json()["detail"]


def test_zone_not_registered_at_revision(client):
    register_square(client, zone_id="px-z")
    _upload(client, "T1", TRACK_X40)
    _upload(client, "T2", TRACK_CONV)
    r = _ask(client, "px-z", revision=999)
    assert r.status_code == 404
    assert "超出当前范围" in r.json()["detail"]
    register_square(client, zone_id="other-zone")  # rev2
    r = _ask(client, "other-zone", revision=1)
    assert r.status_code == 404
    assert "尚未登记" in r.json()["detail"]


def test_trip_not_found(client):
    register_square(client, zone_id="px-t")
    _upload(client, "T1", TRACK_X40)
    # 未指定版本：车次不存在
    r = _ask(client, "px-t", trip_ids=["T1", "ghost"])
    assert r.status_code == 404
    assert "车次不存在" in r.json()["detail"]
    # 指定版本：车次不存在（无任何已保留的轨迹版本）
    r = _ask(client, "px-t", trip_ids=["T1", "ghost"], track_versions={"ghost": 1})
    assert r.status_code == 404
    assert "车次不存在" in r.json()["detail"]


def test_track_version_out_of_range(client):
    register_square(client, zone_id="px-v")
    _upload(client, "T1", TRACK_X40)
    _upload(client, "T2", TRACK_CONV)
    # 早于基线（新车次基线为 1）
    r = _ask(client, "px-v", track_versions={"T1": 0})
    assert r.status_code == 404
    assert "基线" in r.json()["detail"]
    # 晚于当前
    r = _ask(client, "px-v", track_versions={"T2": 9})
    assert r.status_code == 404
    assert "晚于" in r.json()["detail"]


# ---------------------------------------------------------------------
# 错误：请求非法（阈值 / 时间范围 / 重复车次 / 版本格式）
# ---------------------------------------------------------------------

def test_invalid_max_distance_rejected(client):
    register_square(client, zone_id="px-d")
    _upload(client, "T1", TRACK_X40)
    _upload(client, "T2", TRACK_CONV)
    for bad in (0, -5, "NaN", "Infinity", "-Infinity"):
        r = _ask(client, "px-d", max_distance=bad)
        assert r.status_code in (400, 422), bad
    r = _ask(client, "px-d", max_distance="abc")
    assert r.status_code in (400, 422)
    # 缺字段
    r = client.post("/api/v1/zones/px-d/proximity", json={
        "trip_ids": ["T1", "T2"],
        "since": "2026-09-30T10:00:00Z",
        "until": "2026-09-30T10:05:00Z",
    })
    assert r.status_code == 422


def test_duplicate_trip_ids_rejected(client):
    register_square(client, zone_id="px-dup")
    _upload(client, "T1", TRACK_X40)
    r = _ask(client, "px-dup", trip_ids=["T1", "T1"])
    assert r.status_code == 400
    assert "重复" in r.json()["detail"]


def test_invalid_time_range_rejected(client):
    register_square(client, zone_id="px-r")
    _upload(client, "T1", TRACK_X40)
    _upload(client, "T2", TRACK_CONV)
    # since >= until
    r = _ask(client, "px-r", since="2026-09-30T10:02:00Z", until="2026-09-30T10:02:00Z")
    assert r.status_code == 400
    # 不带时区
    r = _ask(client, "px-r", since="2026-09-30T10:00:00", until="2026-09-30T10:02:00Z")
    assert r.status_code == 400
    assert "时区" in r.json()["detail"]
    # 非时间串
    r = _ask(client, "px-r", since="not-a-time")
    assert r.status_code == 400


def test_invalid_request_shape_rejected(client):
    register_square(client, zone_id="px-s")
    # 车次不足两辆
    r = _ask(client, "px-s", trip_ids=["T1"])
    assert r.status_code in (400, 422)
    # 缺字段
    r = client.post("/api/v1/zones/px-s/proximity", json={"trip_ids": ["T1", "T2"]})
    assert r.status_code == 422
    # 多余字段
    r = _ask(client, "px-s", bogus=1)
    assert r.status_code == 422


def test_invalid_track_versions_rejected(client):
    register_square(client, zone_id="px-tv")
    _upload(client, "T1", TRACK_X40)
    _upload(client, "T2", TRACK_CONV)
    # 版本号负数
    r = _ask(client, "px-tv", track_versions={"T1": -1})
    assert r.status_code in (400, 422)
    # 版本号非整数
    r = _ask(client, "px-tv", track_versions={"T1": "abc"})
    assert r.status_code in (400, 422)
    # 键不在 trip_ids 中
    r = _ask(client, "px-tv", track_versions={"T9": 1})
    assert r.status_code == 400
    assert "T9" in r.json()["detail"]


# ---------------------------------------------------------------------
# 同一已提交读视图：并发更新不拼出不存在的组合
# ---------------------------------------------------------------------

def test_concurrent_writes_never_break_snapshot(client):
    """writer 持续更新区域与轨迹，reader 反复查询：
    结果必须结构完整；固定版本组合的复现结果确定一致。"""
    register_square(client, zone_id="px-live")
    _upload(client, "T1", TRACK_X40)
    _upload(client, "T2", TRACK_CONV)

    stop = threading.Event()
    failures = []

    def writer() -> None:
        i = 0
        while not stop.is_set():
            size = 100 + (i % 3) * 50
            client.post(
                "/api/v1/zones",
                json={
                    "zone_id": "px-live",
                    "exterior": [[0, 0], [size, 0], [size, size], [0, size]],
                    "holes": [],
                    "windows": DAY_WINDOWS,
                },
            )
            client.post(
                "/api/v1/trips/T1/points",
                json={"points": [
                    {"point_id": f"extra-{i}", "ts": f"2026-09-30T12:{i % 60:02d}:00Z",
                     "x": 10 + i, "y": 10},
                ]},
            )
            i += 1

    def reader() -> None:
        while not stop.is_set():
            r = _ask(client, "px-live")
            if r.status_code != 200:
                failures.append(("status", r.status_code, r.text))
                return
            body = r.json()
            trips = {t["trip_id"]: t for t in body["trips"]}
            # 回填的版本与点数必须来自同一快照：版本单调推进、点数与版本一致
            if trips["T1"]["track_version"] < 1 or trips["T2"]["track_version"] != 1:
                failures.append(("version-mix", body))
                return
            for pair in body["pairs"]:
                if len(pair["trip_ids"]) != 2 or pair["trip_ids"][0] == pair["trip_ids"][1]:
                    failures.append(("pair", pair))
                    return
                for seg in pair["intervals"]:
                    if seg["duration_sec"] <= 0 or not (0 <= seg["min_distance"] < 25):
                        failures.append(("segment", seg))
                        return
            # 固定组合复现：rev1 区域 × T1@1/T2@1，两次结果必须一致
            pinned = dict(revision=1, track_versions={"T1": 1, "T2": 1})
            r1 = _ask(client, "px-live", **pinned)
            r2 = _ask(client, "px-live", **pinned)
            if r1.status_code != 200 or r1.json() != r2.json():
                failures.append(("pinned-nondeterministic", r1.status_code, r1.text))
                return
            seg = r1.json()["pairs"][0]["intervals"][0]
            if seg["duration_sec"] != 37.5 or seg["min_distance"] != 10.0:
                failures.append(("pinned-wrong", r1.json()))
                return

    threads = [threading.Thread(target=writer)] + [threading.Thread(target=reader) for _ in range(2)]
    for t in threads:
        t.start()
    stop.wait(timeout=1.0)
    stop.set()
    for t in threads:
        t.join(timeout=10)

    assert not failures


def test_existing_endpoints_unaffected(client):
    """近距离相遇查询上线后，既有穿越与多车占用查询语义不变。"""
    register_square(client, zone_id="px-keep")
    _upload(client, "T1", TRACK_X40)
    _upload(client, "T2", TRACK_CONV)
    r = client.get("/api/v1/trips/T1/crossings?zone_id=px-keep")
    assert r.status_code == 200
    z = r.json()["zones"][0]
    assert z["total_duration_sec"] == 100.0
    assert "track_version" not in r.json()

    r = client.post("/api/v1/zones/px-keep/co-occupancy", json={
        "trip_ids": ["T1", "T2"],
        "since": "2026-09-30T10:00:00Z",
        "until": "2026-09-30T10:05:00Z",
        "threshold": 2,
    })
    assert r.status_code == 200
    assert len(r.json()["intervals"]) == 1
