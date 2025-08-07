

def soft_overlong_punishment(response_length, max_length, length_cache=512):
    if response_length <= max_length - length_cache:
        return 0.0

    if max_length - length_cache < response_length <= max_length:
        return float((max_length - length_cache) - response_length) / float(length_cache)

    if response_length > max_length:
        return -1.0