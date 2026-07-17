from aiperf.dataset.graph.segment_ir.pool import SegmentPool


def test_rolling_id_is_prefix_dependent_and_deterministic():
    pool = SegmentPool()
    a = pool.add(role="system", content="sys", tokens=[1, 2], parent_id=None)
    b1 = pool.add(role="user", content="u", tokens=[3], parent_id=a)
    # same content+prefix -> same id (dedup)
    b2 = pool.add(role="user", content="u", tokens=[3], parent_id=a)
    assert b1 == b2
    # same content, DIFFERENT prefix -> different id (rolling)
    c = pool.add(role="system", content="other", tokens=[9], parent_id=None)
    b3 = pool.add(role="user", content="u", tokens=[3], parent_id=c)
    assert b3 != b1


def test_materialize_path_is_exact_messages_in_order():
    pool = SegmentPool()
    a = pool.add(role="system", content="sys", tokens=[1], parent_id=None)
    b = pool.add(role="user", content="hi", tokens=[2], parent_id=a)
    d = pool.add(role="assistant", content="yo", tokens=[3], parent_id=b)
    assert pool.materialize([a, b, d]) == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]


def test_truncation_branches_at_shared_prefix():
    pool = SegmentPool()
    s1 = pool.add(role="system", content="1", tokens=[1], parent_id=None)
    s2 = pool.add(role="user", content="2", tokens=[2], parent_id=s1)
    s3 = pool.add(role="assistant", content="3", tokens=[3], parent_id=s2)
    # truncation reuses [s1,s2], drops s3, adds s5 off s2
    s5 = pool.add(role="user", content="5", tokens=[5], parent_id=s2)
    assert s5 != s3 and pool.get(s5).parent_id == s2
