from src.metrics.communication import fp32_uncompressed_payload_bits, metadata_bits, payload_bits


def test_payload_bits_use_group_widths():
    nvals = {0: 100, 1: 200}
    widths = {0: 2, 1: 8}
    assert payload_bits(nvals, widths) == (100 * 2 + 200 * 8)


def test_metadata_bits_separate_from_payload():
    meta = metadata_bits(
        [2, 4, 8, 8],
        group_to_channels={0: [0, 1], 1: [2, 3]},
        group_bit_widths={0: 2, 1: 8},
    )
    assert meta > 0


def test_fp32_payload_bits():
    assert fp32_uncompressed_payload_bits(10) == 320
