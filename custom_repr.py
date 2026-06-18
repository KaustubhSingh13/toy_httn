import numpy as np


def fmt_array(arr, first_line_offset):
    rows = ["[" + ", ".join(f"{x:g}" for x in row) + "]" for row in arr]
    row_pad = " " * (first_line_offset + 1)
    inner = (",\n" + row_pad).join(rows)
    result = f"[{inner}]"
    if len(arr) == 1:
        last_line_len = first_line_offset + 1 + len(rows[0]) + 1
    else:
        last_line_len = len(row_pad) + len(rows[-1]) + 1
    return result, last_line_len


def _is_dictlike(obj):
    return isinstance(obj, dict) or hasattr(obj, 'site_ops')


def _fmt_term(d, inner_offset):
    """Format a single term inline. inner_offset = col right after the term's '{'."""
    all_pairs = sorted(d.items(), key=lambda x: x[0])
    chunks = [all_pairs[i:i+4] for i in range(0, len(all_pairs), 4)]

    dict_str = ""
    col = inner_offset

    for chunk_idx, chunk in enumerate(chunks):
        if chunk_idx == 0:
            line = ""
        else:
            line = " " * inner_offset
            col = inner_offset

        for idx, (k, v) in enumerate(chunk):
            if idx > 0:
                line += ", "
                col += 2

            key_part = f"{k!r}: "
            line += key_part
            col += len(key_part)

            if isinstance(v, np.ndarray):
                val_str, last_line_len = fmt_array(v, col)
                line += val_str
                col = last_line_len
            else:
                val_str = _repr_value(v, _line_offset=col)
                line += val_str
                col += len(val_str)

        dict_str += ("\n" if chunk_idx > 0 else "") + line

    return dict_str, col


def _repr_value(obj, indent=0, _line_offset=0):
    pad = "  " * indent

    if _is_dictlike(obj):
        lines = []
        for k, v in obj.items():
            key_str = f"{pad}{k!r}: "
            val_str = _repr_value(v, indent + 1, _line_offset=len(key_str))
            lines.append(key_str + val_str)
        return "{\n" + ",\n".join(lines) + "\n" + pad + "}"

    elif isinstance(obj, set) and all(_is_dictlike(i) for i in obj):
        # Each term gets its own line, all at the same column as the first term
        # First term starts at _line_offset+1 (right after outer "{")
        term_indent = _line_offset + 1
        term_pad = " " * term_indent

        dicts = sorted(obj, key=lambda d: min(d.keys()))
        parts = []
        for d in dicts:
            body, _ = _fmt_term(d, inner_offset=term_indent + 1)
            parts.append("{" + body + "}")

        # First term joins directly after "{", rest go to new lines at term_indent
        return "{" + (",\n" + term_pad).join(parts) + "}"

    elif isinstance(obj, np.ndarray):
        result, _ = fmt_array(obj, _line_offset)
        return result

    else:
        return repr(obj)
