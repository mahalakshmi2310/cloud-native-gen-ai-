def create_payload(df, seq_len):
    if len(df) < seq_len:
        raise ValueError("Not enough rows")
    tail = df.tail(seq_len)
    return tail.to_dict(orient="records"), tail
