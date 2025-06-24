from predict import load_data


def test_data_loads():
    df = load_data()
    assert not df.empty
