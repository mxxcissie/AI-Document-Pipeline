def chunk_text(text: str, chunk_size: int = 300, chunk_overlap: int = 50) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            while end > start and text[end] != " ":
                end -= 1
            if end == start:
                end = min(start + chunk_size, text_length)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = max(end - chunk_overlap, 0)

        while next_start < text_length and text[next_start] != " ":
            next_start += 1
        while next_start < text_length and text[next_start] == " ":
            next_start += 1

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks